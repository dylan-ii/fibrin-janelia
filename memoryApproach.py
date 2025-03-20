import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tifffile
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ------------------------------------------------------------------
# Dataset: Load binary 3D TIF volumes and their masks with downsampling
# ------------------------------------------------------------------
class FiberTiffDataset(Dataset):
    """
    Loads 3D TIF volumes from data_dir (raw volumes) and mask_dir (ground truth),
    returning (memory_volumes, memory_masks, query_volume, query_mask).

    Each TIF is assumed to be a 3D stack of shape [D, H, W]. The data is binary.
    """
    def __init__(self,
                 data_dir,
                 mask_dir,
                 n_memory=2,
                 downsample_factor=2,
                 transform=None):
        """
        Args:
            data_dir (str): Path to directory of raw 3D TIF files (time steps).
            mask_dir (str): Path to directory of 3D TIF ground-truth masks.
            n_memory (int): Number of previous frames to use as memory.
            downsample_factor (int): Factor to downsample each dimension.
            transform (callable, optional): Optional transform to apply after loading.
        """
        super().__init__()
        self.data_dir = data_dir
        self.mask_dir = mask_dir
        self.n_memory = n_memory
        self.downsample_factor = downsample_factor
        self.transform = transform

        # list TIF files
        self.data_files = sorted([
            f for f in os.listdir(data_dir)
            if f.lower().endswith('.tif') or f.lower().endswith('.tiff')
        ])
        self.mask_files = sorted([
            f for f in os.listdir(mask_dir)
            if f.lower().endswith('.tif') or f.lower().endswith('.tiff')
        ])

        if len(self.data_files) != len(self.mask_files):
            raise ValueError("Number of data TIFs and mask TIFs must match.")

        self.valid_indices = list(range(self.n_memory, len(self.data_files)))
        print(f"Dataset initialized with {len(self.valid_indices)} valid samples.")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        t = self.valid_indices[idx]

        # load query volume and its mask
        query_volume = self._load_tif(os.path.join(self.data_dir, self.data_files[t]))
        query_mask   = self._load_tif(os.path.join(self.mask_dir,  self.mask_files[t]))

        # collect memory volumes and masks
        memory_volumes = []
        memory_masks   = []
        for offset in range(1, self.n_memory+1):
            mem_t = t - offset
            mem_vol = self._load_tif(os.path.join(self.data_dir, self.data_files[mem_t]))
            mem_mask = self._load_tif(os.path.join(self.mask_dir,  self.mask_files[mem_t]))
            memory_volumes.append(mem_vol)
            memory_masks.append(mem_mask)

        memory_volumes.reverse()
        memory_masks.reverse()

        if self.transform is not None:
            query_volume, query_mask = self.transform(query_volume, query_mask)

        return {
            'memory_volumes': memory_volumes,  # list of Tensors, each [1, D', H', W']
            'memory_masks':   memory_masks,    # list of Tensors, each [1, D', H', W']
            'query_volume':   query_volume,    # Tensor [1, D', H', W']
            'query_mask':     query_mask       # Tensor [1, D', H', W']
        }

    def _load_tif(self, path):
        """
        Loads a 3D TIF file as a PyTorch float tensor of shape [1, D, H, W],
        then downsamples it by self.downsample_factor using nearest neighbor.
        Assumes the TIF is binary or single-channel grayscale.
        """
        vol = tifffile.imread(path)
        vol = torch.from_numpy(vol).float()

        vol = torch.clamp(vol, 0, 1)
        vol = vol.unsqueeze(0)

        if self.downsample_factor != 1:
            df = self.downsample_factor
            vol = F.interpolate(
                vol.unsqueeze(0),
                scale_factor=(1/df, 1/df, 1/df),
                mode='nearest'
            ).squeeze(0)

        return vol

# ------------------------------------------------------------------
# Model Components
# ------------------------------------------------------------------
class Simple3DEncoder(nn.Module):
    def __init__(self, in_channels=1, feature_dim=64):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm3d(32)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(32, feature_dim, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm3d(feature_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class MemoryEncoder3D(nn.Module):
    def __init__(self, feature_dim=64, num_objects=1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_objects = num_objects

    def forward(self, memory_feat, memory_mask):
        """
        memory_feat: [B, feature_dim, D, H, W]
        memory_mask: [B, num_objects, D, H, W]
        """
        B, feat_dim, D, H, W = memory_feat.shape
        object_embeddings = []
        for obj_idx in range(self.num_objects):
            mask = memory_mask[:, obj_idx] 
            mask_exp = mask.unsqueeze(1) 
            masked_feat = memory_feat * mask_exp
            sum_feat = masked_feat.view(B, feat_dim, -1).sum(dim=-1)
            sum_mask = mask.view(B, -1).sum(dim=-1).clamp(min=1e-6)
            mean_feat = sum_feat / sum_mask.unsqueeze(-1)
            object_embeddings.append(mean_feat)
        object_embeddings = torch.stack(object_embeddings, dim=1)
        pixel_memory = memory_feat
        return pixel_memory, object_embeddings

class TransformerBlock3D(nn.Module):
    def __init__(self, d_model=64, n_heads=8, dim_feedforward=256):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, obj_emb, pixel_emb):
        obj_emb2, _ = self.self_attn(obj_emb, obj_emb, obj_emb)
        obj_emb = self.norm1(obj_emb + self.dropout(obj_emb2))
        obj_emb2, _ = self.cross_attn(obj_emb, pixel_emb, pixel_emb)
        obj_emb = self.norm2(obj_emb + self.dropout(obj_emb2))
        ff = self.linear2(self.relu(self.linear1(obj_emb)))
        obj_emb = self.norm3(obj_emb + self.dropout(ff))
        return obj_emb

class ObjectTransformer3D(nn.Module):
    def __init__(self, d_model=64, n_heads=8, dim_feedforward=256, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock3D(d_model, n_heads, dim_feedforward)
            for _ in range(num_layers)
        ])

    def forward(self, obj_emb, pixel_emb):
        for layer in self.layers:
            obj_emb = layer(obj_emb, pixel_emb)
        return obj_emb

class Decoder3D(nn.Module):
    def __init__(self, feature_dim=64, num_objects=1):
        super().__init__()
        self.num_objects = num_objects
        self.conv = nn.Conv3d(feature_dim, feature_dim, kernel_size=3, padding=1)
        self.bn   = nn.BatchNorm3d(feature_dim)
        self.relu = nn.ReLU(inplace=True)
        self.classifier = nn.Conv3d(feature_dim, num_objects, kernel_size=1)

    def forward(self, query_feat, obj_emb):
        """
        query_feat: [B, feature_dim, D, H, W]
        obj_emb: [B, num_objects, feature_dim]
        returns: [B, num_objects, D, H, W]
        """
        B, feat_dim, D, H, W = query_feat.shape
        obj_emb_expanded = obj_emb.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        obj_emb_expanded = obj_emb_expanded.expand(-1, -1, -1, D, H, W)
        masks = []
        for obj_idx in range(self.num_objects):
            emb = obj_emb_expanded[:, obj_idx]
            combined = query_feat + emb
            x = self.conv(combined)
            x = self.bn(x)
            x = self.relu(x)
            logit = self.classifier(x)
            single_obj_logit = logit[:, 0:1]
            masks.append(single_obj_logit)
        masks = torch.cat(masks, dim=1)
        return masks

class FiberTrackingModel3D(nn.Module):
    def __init__(self, 
                 in_channels=1,
                 feature_dim=64,
                 num_objects=1,
                 num_transformer_layers=4):
        super().__init__()
        self.num_objects = num_objects
        self.query_encoder = Simple3DEncoder(in_channels, feature_dim)
        self.memory_encoder = Simple3DEncoder(in_channels, feature_dim)
        self.mask_encoder = MemoryEncoder3D(feature_dim, num_objects)
        self.object_transformer = ObjectTransformer3D(
            d_model=feature_dim,
            n_heads=8,
            dim_feedforward=256,
            num_layers=num_transformer_layers
        )
        self.decoder = Decoder3D(feature_dim, num_objects)

    def forward(self, memory_volumes, memory_masks, query_volume):
        """
        memory_volumes: list of [B, 1, D, H, W]
        memory_masks:   list of [B, num_objects, D, H, W]
        query_volume:   [B, 1, D, H, W]
        """
        B = query_volume.shape[0]
        query_feat = self.query_encoder(query_volume)
        D, H, W = query_feat.shape[-3:]
        query_feat_flat = query_feat.view(B, -1, D*H*W).permute(0, 2, 1)

        pixel_memories = []
        object_memories = []
        for mem_vol, mem_mask in zip(memory_volumes, memory_masks):
            mem_feat = self.memory_encoder(mem_vol)
            pixel_memory, object_memory = self.mask_encoder(mem_feat, mem_mask)
            pixel_memories.append(pixel_memory)
            object_memories.append(object_memory)

        pixel_memory_flat_list = []
        for pm in pixel_memories:
            pm_flat = pm.view(B, -1, D*H*W).permute(0, 2, 1)
            pixel_memory_flat_list.append(pm_flat)

        if len(pixel_memory_flat_list) > 0:
            pixel_memory_cat = torch.cat(pixel_memory_flat_list, dim=1)
        else:
            pixel_memory_cat = query_feat_flat

        if len(object_memories) > 0:
            object_memory_cat = torch.mean(torch.stack(object_memories, dim=0), dim=0)
        else:
            object_memory_cat = torch.zeros(B, self.num_objects, query_feat.shape[1], device=query_feat.device)

        refined_obj_emb = self.object_transformer(object_memory_cat, pixel_memory_cat)
        pred_masks = self.decoder(query_feat, refined_obj_emb)
        return pred_masks

# ------------------------------------------------------------------
# Training Helpers
# ------------------------------------------------------------------
def dice_loss(pred, target, eps=1e-6):
    """
    pred: [B, num_objects, D, H, W] (logits)
    target: [B, num_objects, D, H, W] (binary)
    """
    pred_sig = torch.sigmoid(pred)
    intersection = (pred_sig * target).sum(dim=[2,3,4])
    union = pred_sig.sum(dim=[2,3,4]) + target.sum(dim=[2,3,4]) + eps
    dice = 2.0 * intersection / union
    return 1.0 - dice.mean()

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    start_time = time.time()
    for i, batch in enumerate(tqdm(dataloader, desc="Training Batches")):
        memory_volumes = [v.to(device) for v in batch['memory_volumes']]
        memory_masks   = [m.to(device) for m in batch['memory_masks']]
        query_volume   = batch['query_volume'].to(device)
        query_mask     = batch['query_mask'].to(device)

        optimizer.zero_grad()
        pred_masks = model(memory_volumes, memory_masks, query_volume)
        loss = dice_loss(pred_masks, query_mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % 10 == 0 or (i + 1) == num_batches:
            print(f"  Batch {i+1}/{num_batches} - Loss: {loss.item():.4f}")

    elapsed = time.time() - start_time
    print(f"Training epoch completed in {elapsed:.2f} seconds.")
    return total_loss / num_batches

def validate_one_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)
    start_time = time.time()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Validation Batches")):
            memory_volumes = [v.to(device) for v in batch['memory_volumes']]
            memory_masks   = [m.to(device) for m in batch['memory_masks']]
            query_volume   = batch['query_volume'].to(device)
            query_mask     = batch['query_mask'].to(device)

            pred_masks = model(memory_volumes, memory_masks, query_volume)
            loss = dice_loss(pred_masks, query_mask)
            total_loss += loss.item()
            if (i + 1) % 10 == 0 or (i + 1) == num_batches:
                print(f"  Batch {i+1}/{num_batches} - Loss: {loss.item():.4f}")
    elapsed = time.time() - start_time
    print(f"Validation epoch completed in {elapsed:.2f} seconds.")
    return total_loss / num_batches

# ------------------------------------------------------------------
# Main Training Loop with Progress Logging
# ------------------------------------------------------------------
def main():
    data_dir = "20241024/binary_volumes"
    mask_dir = "20241024/labeled_volumes"
    n_memory = 2
    downsample_factor = 4 
    batch_size = 1
    num_workers = 4
    num_epochs = 10
    lr = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    full_dataset = FiberTiffDataset(
        data_dir=data_dir,
        mask_dir=mask_dir,
        n_memory=n_memory,
        downsample_factor=downsample_factor
    )

    # (80/20 split)
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    print(f"Training samples: {train_size}, Validation samples: {val_size}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # initialize model, optimizer
    model = FiberTrackingModel3D(
        in_channels=1,
        feature_dim=64,
        num_objects=1,            # only one fiber class
        num_transformer_layers=2  # reduced for speed
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        epoch_start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss   = validate_one_epoch(model, val_loader, device)
        epoch_time = time.time() - epoch_start
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Duration: {epoch_time:.2f} sec")

    model.eval()
    with torch.no_grad():
        batch = next(iter(val_loader))
        memory_volumes = [v.to(device) for v in batch['memory_volumes']]
        memory_masks   = [m.to(device) for m in batch['memory_masks']]
        query_volume   = batch['query_volume'].to(device)

        pred_masks = model(memory_volumes, memory_masks, query_volume)
        print("Predicted mask shape:", pred_masks.shape)
        pred_binary = (torch.sigmoid(pred_masks) > 0.5).float()
        print("Example output (binary mask):", pred_binary[0,0])

if __name__ == "__main__":
    main()
