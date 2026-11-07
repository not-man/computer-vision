import numpy as np
import torch

from mct.adapt import CHAINS_PER_BATCH, PER_CHAIN, ChainSampler, augment, masked_contrastive


def unit(*rows: list[float]) -> torch.Tensor:
    return torch.nn.functional.normalize(torch.tensor(rows, dtype=torch.float32), dim=1)


def table(size: int, *pairs: tuple[int, int]) -> torch.Tensor:
    different = torch.zeros(size, size, dtype=torch.bool)
    for a, b in pairs:
        different[a, b] = different[b, a] = True
    return different


def test_loss_falls_when_own_views_come_closer_and_strangers_move_away() -> None:
    chains = torch.tensor([0, 0, 1])
    far = masked_contrastive(unit([1, 0], [0, 1], [1, 0.1]), chains, table(2, (0, 1)))
    near = masked_contrastive(unit([1, 0], [1, 0.05], [0, 1]), chains, table(2, (0, 1)))
    assert near < far


def test_person_with_unknown_relation_does_not_change_the_loss() -> None:
    # Третья цепочка с неизвестным отношением может быть тем же человеком: потеря её не видит, как бы близко она ни лежала
    known = masked_contrastive(unit([1, 0], [1, 0.3], [0, 1]), torch.tensor([0, 0, 1]), table(3, (0, 1)))
    extra = masked_contrastive(unit([1, 0], [1, 0.3], [0, 1], [1, 0.1]), torch.tensor([0, 0, 1, 2]), table(3, (0, 1)))
    assert abs(float(known) - float(extra)) < 1e-5


def test_no_known_pairs_give_zero_loss_and_a_usable_gradient() -> None:
    vectors = unit([1, 0], [0, 1]).requires_grad_(True)
    loss = masked_contrastive(vectors, torch.tensor([0, 1]), table(2))
    loss.backward()
    assert float(loss.detach()) == 0.0 and torch.isfinite(vectors.grad).all()


def test_gradient_stays_finite_when_only_some_anchors_have_strangers() -> None:
    vectors = unit([1, 0], [1, 0.2], [0, 1], [0.5, 0.5], [0.4, 0.6]).requires_grad_(True)
    masked_contrastive(vectors, torch.tensor([0, 0, 1, 2, 2]), table(3, (0, 1))).backward()
    assert torch.isfinite(vectors.grad).all()


def test_sampler_takes_strangers_of_the_first_chain_and_prefers_other_cameras() -> None:
    chains = np.array([0, 0, 0, 0, 1, 1, 2, 2, -1])
    cameras = np.array(["a", "a", "b", "b", "a", "a", "c", "c", "a"])
    different = np.zeros((3, 3), bool)
    different[0, 1] = different[1, 0] = True
    sampler = ChainSampler(chains, cameras, different, seed=1)
    for _ in range(20):
        rows = sampler.batch()
        assert 8 not in rows  # снимок без подписи
        assert len(rows) <= CHAINS_PER_BATCH * PER_CHAIN
        own = rows[chains[rows] == 0]
        assert set(chains[rows]) >= {0, 1}
        assert set(cameras[own]) == {"a", "b"}


def test_augment_keeps_shape_and_range() -> None:
    images = torch.rand(6, 3, 256, 128)
    result = augment(images.clone())
    assert result.shape == images.shape and result.min() >= 0 and result.max() <= 1
