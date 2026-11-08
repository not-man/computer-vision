"""Дообучение модели внешности на самой записи по подписям из геометрии и времени.

Потери считают только известные пары: свои и заведомо чужие. Остальные пары не участвуют: вернувшийся человек с другой цепочкой
иначе учился бы быть непохожим на себя, а именно таких людей модель и должна узнавать.
"""
import numpy as np
import torch
import torch.nn.functional as F

INPUT_SIZE = (128, 256)  # ширина и высота входа OSNet
MEAN = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
TEMPERATURE = 0.1
CHAINS_PER_BATCH = 32
PER_CHAIN = 4
LEARNING_RATE = 2e-4
EARLY_LAYERS_SLOWER = 0.1
WEIGHT_DECAY = 5e-4


def masked_contrastive(vectors: torch.Tensor, chains: torch.Tensor, different: torch.Tensor) -> torch.Tensor:
    """Каждая своя пара против заведомо чужих того же снимка. different: квадратная таблица цепочек."""
    same = chains[:, None] == chains[None, :]
    same.fill_diagonal_(False)
    other = different[chains[:, None], chains[None, :]]
    anchors = same.any(dim=1) & other.any(dim=1)
    if not anchors.any():
        return vectors.sum() * 0.0
    similarity = (vectors @ vectors.T / TEMPERATURE)[anchors]
    rivals = torch.logsumexp(similarity.masked_fill(~other[anchors], float("-inf")), dim=1, keepdim=True)
    return -(similarity - torch.logaddexp(similarity, rivals))[same[anchors]].mean()


class ChainSampler:
    """Снимки для шага обучения: цепочка, заведомо чужие ей и случайные, у каждой снимки по возможности с разных камер."""

    def __init__(self, chains: np.ndarray, cameras: np.ndarray, different: np.ndarray, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.by_chain: dict[int, dict[str, np.ndarray]] = {}
        for chain in np.unique(chains[chains >= 0]):
            rows = np.flatnonzero(chains == chain)
            self.by_chain[int(chain)] = {camera: rows[cameras[rows] == camera] for camera in np.unique(cameras[rows])}
        self.all = sorted(self.by_chain)
        self.several = [chain for chain in self.all if len(self.by_chain[chain]) > 1] or self.all
        self.rivals = {chain: np.flatnonzero(different[chain]).tolist() for chain in self.all}

    def batch(self) -> np.ndarray:
        first = int(self.rng.choice(self.several))
        rivals = [chain for chain in self.rivals[first] if chain in self.by_chain]
        chosen = [first] + [int(chain) for chain in self.rng.permutation(rivals)[:CHAINS_PER_BATCH // 2]]
        rest = [chain for chain in self.rng.permutation(self.all) if int(chain) not in chosen]
        chosen += [int(chain) for chain in rest[:CHAINS_PER_BATCH - len(chosen)]]
        rows = []
        for chain in chosen:
            cameras = list(self.by_chain[chain].values())
            order = self.rng.permutation(len(cameras))
            rows += [int(self.rng.choice(cameras[order[i % len(cameras)]])) for i in range(PER_CHAIN)]
        return np.array(rows)


def augment(images: torch.Tensor) -> torch.Tensor:
    """Вход 0..1. Сдвиг, масштаб и отражение; цветной свет; потеря резкости как вдали; закрытая часть человека."""
    count, device = len(images), images.device
    scale = torch.empty(count, device=device).uniform_(0.85, 1.0)
    flip = torch.where(torch.rand(count, device=device) < 0.5, -1.0, 1.0)
    theta = torch.zeros(count, 2, 3, device=device)
    theta[:, 0, 0], theta[:, 1, 1] = scale * flip, scale
    theta[:, 0, 2] = (1 - scale) * torch.empty(count, device=device).uniform_(-1, 1)
    theta[:, 1, 2] = (1 - scale) * torch.empty(count, device=device).uniform_(-1, 1)
    grid = F.affine_grid(theta, images.shape, align_corners=False)
    images = F.grid_sample(images, grid, mode="bilinear", padding_mode="border", align_corners=False)
    images = (images * torch.empty(count, 3, 1, 1, device=device).uniform_(0.7, 1.3)).clamp(0, 1)

    height, width = images.shape[2:]
    factor = float(torch.empty(1).uniform_(0.25, 0.6))
    blurred = F.interpolate(F.interpolate(images, scale_factor=factor, mode="area"), size=(height, width), mode="bilinear", align_corners=False)
    images = torch.where(torch.rand(count, 1, 1, 1, device=device) < 0.5, blurred, images)
    for row in torch.nonzero(torch.rand(count) < 0.5).flatten().tolist():
        h, w = int(height * float(torch.empty(1).uniform_(0.1, 0.4))), int(width * float(torch.empty(1).uniform_(0.2, 0.6)))
        y, x = int(torch.randint(0, height - h + 1, (1,))), int(torch.randint(0, width - w + 1, (1,)))
        images[row, :, y:y + h, x:x + w] = 0.5
    return images


def features(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Нормированные векторы как в обычной работе модели, но с градиентом и в режиме обучения. Вход 0..1."""
    normal = (images - MEAN.to(images.device)) / STD.to(images.device)
    vectors = model.global_avgpool(model.forward_features(normal)).flatten(1)
    return F.normalize(model.fc(vectors) if model.fc is not None else vectors, dim=1)


def embed_all(model: torch.nn.Module, images: torch.Tensor, device: str, batch_size: int = 512) -> np.ndarray:
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            parts.append(features(model, images[start:start + batch_size].to(device).float() / 255).cpu().numpy())
    model.train()
    return np.vstack(parts)


def recalibrate_norms(model: torch.nn.Module, images: torch.Tensor, device: str, batches: int = 200, batch_size: int = 128, seed: int = 0) -> None:
    """Обновляет статистику нормировок на кадрах этой записи без обучения весов."""
    rng = np.random.default_rng(seed)
    norms = [layer for layer in model.modules() if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm)]
    speeds = [layer.momentum for layer in norms]
    for layer in norms:
        layer.reset_running_stats()
        layer.momentum = None  # простое среднее по всем шагам
    model.train()
    with torch.no_grad():
        for _ in range(batches):
            rows = rng.integers(0, len(images), batch_size)
            features(model, images[rows].to(device).float() / 255)
    for layer, speed in zip(norms, speeds):
        layer.momentum = speed


def train(model: torch.nn.Module, images: torch.Tensor, sampler: ChainSampler, different: torch.Tensor, chains: torch.Tensor, steps: int, device: str, on_checkpoint, checkpoints: tuple[int, ...]) -> None:
    late = [*model.conv4.parameters(), *model.conv5.parameters(), *(model.fc.parameters() if model.fc is not None else [])]
    late_ids = {id(parameter) for parameter in late}
    early = [parameter for name, parameter in model.named_parameters() if id(parameter) not in late_ids and not name.startswith("classifier")]
    optimizer = torch.optim.AdamW([{"params": early, "lr": LEARNING_RATE * EARLY_LAYERS_SLOWER}, {"params": late, "lr": LEARNING_RATE}], weight_decay=WEIGHT_DECAY)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device != "cpu")
    model.requires_grad_(True)
    model.train()
    different = different.to(device)
    for step in range(1, steps + 1):
        rows = torch.as_tensor(sampler.batch())
        batch = augment(images[rows].to(device).float() / 255)
        with torch.autocast("cuda", enabled=device != "cpu"):
            vectors = features(model, batch)
        loss = masked_contrastive(vectors.float(), chains[rows].to(device), different)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        schedule.step()
        if step % 100 == 0:
            print(f"шаг {step} из {steps}, потеря {float(loss):.3f}", flush=True)
        if step in checkpoints:
            on_checkpoint(step)
