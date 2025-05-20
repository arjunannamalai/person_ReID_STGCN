import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import average_precision_score
from scipy.spatial.distance import cdist
from PIL import Image
from collections import defaultdict
import time
from torch.cuda.amp import GradScaler, autocast

# Configuration
class Config:
    seed = 42
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_paths = {
        "train": "/scratch/cs25s007/dlp/MSMT17/bounding_box_train",
        "test": "reidgpt/dataset/MSMT17/bounding_box_test",
        "query": "/scratch/cs25s007/dlp/MSMT17/query"
    }
    model_dir = "./models"
    checkpoint_dir = "./checkpoints"
    num_epochs = 1000
    batch_size = 128
    num_workers = 2
    P = 32  # Number of person IDs per batch
    K = 4   # Number of instances per person ID
    k1 = 20  # Re-ranking parameter
    k2 = 6   # Re-ranking parameter
    lambda_value = 0.3  # Re-ranking parameter
    margin = 0.3  # Triplet loss margin

# Initialize directories
os.makedirs(Config.model_dir, exist_ok=True)
os.makedirs(Config.checkpoint_dir, exist_ok=True)

# Set random seeds
random.seed(Config.seed)
np.random.seed(Config.seed)
torch.manual_seed(Config.seed)
if Config.device.type == 'cuda':
    torch.cuda.manual_seed_all(Config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
os.environ["PYTHONHASHSEED"] = str(Config.seed)

# Data Transforms
transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3))
])

# Dataset Class
class MSMT17Dataset(Dataset):
    def __init__(self, root_dir, transform=None, num_frames=8):
        self.root_dir = root_dir
        self.transform = transform
        self.num_frames = num_frames
        self._init_dataset()
        self.cache = {}

    def _init_dataset(self):
        self.tracklets = []
        self.pid_to_indices = defaultdict(list)
        self.unique_pids = set()
        tracklet_dict = defaultdict(list)

        for fn in os.listdir(self.root_dir):
            if not fn.endswith('.jpg'):
                continue
            try:
                parts = os.path.splitext(fn)[0].split('_')
                pid = int(parts[0])
                camseq = parts[1]
                camid = int(camseq.split('s')[0][1:]) if 's' in camseq else int(camseq[1:])
                seqid = int(camseq.split('s')[1]) if 's' in camseq else 0
                tracklet_dict[(pid, camid, seqid)].append(fn)
                self.unique_pids.add(pid)
            except Exception:
                continue

        for (pid, camid, seqid), fns in tracklet_dict.items():
            self.tracklets.append({
                'pid': pid,
                'camid': camid,
                'seqid': seqid,
                'frames': self._sample_frames(fns)
            })
            self.pid_to_indices[pid].append(len(self.tracklets)-1)

        self.unique_pids = sorted(self.unique_pids)

    def _sample_frames(self, fns):
        if len(fns) >= self.num_frames:
            return random.sample(fns, self.num_frames)
        return fns + random.choices(fns, k=self.num_frames-len(fns))

    def _load_image(self, fn):
        if fn not in self.cache:
            img = Image.open(os.path.join(self.root_dir, fn))
            self.cache[fn] = img
        return self.cache[fn]

    def __len__(self):
        return len(self.tracklets)

    def __getitem__(self, idx):
        tracklet = self.tracklets[idx]
        frames = [self.transform(self._load_image(fn)) for fn in tracklet['frames']]
        return torch.stack(frames), tracklet['pid'], tracklet['camid']

# Model Components
class TemporalGCN(nn.Module):
    def __init__(self, in_dim=2048):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.Sigmoid()
        )
        self.gcn = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, in_dim)
        )

    def forward(self, X):
        B, N, C = X.shape
        attn_weights = self.attention(X.mean(1)).unsqueeze(1)
        X = X * attn_weights
        return self.gcn(X.view(-1, C)).view(B, N, C)

class SpatialGCN(nn.Module):
    def __init__(self, in_dim=2048):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.Sigmoid()
        )
        self.gcn = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, in_dim)
        )

    def forward(self, X):
        B, T, P, C = X.shape
        X = X.view(B*T, P, C)
        attn_weights = self.attention(X.mean(1)).unsqueeze(1)
        X = X * attn_weights
        outputs = [self.gcn(frame) for frame in X]
        return torch.stack(outputs).view(B, T, P, C)

# ST-GCN Model
class STGCN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Partial freezing
        for i, child in enumerate(backbone.children()):
            if i < 6:  # Freeze first 6 blocks
                for param in child.parameters():
                    param.requires_grad = False
        
        # Adjust strides
        for block in backbone.layer4.children():
            if isinstance(block, models.resnet.Bottleneck):
                block.conv1.stride = (1, 1)
                block.conv2.stride = (1, 1)
                if block.downsample is not None:
                    block.downsample[0].stride = (1, 1)
        
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.temporal_gcn = TemporalGCN()
        self.spatial_gcn = SpatialGCN()
        self.feature_fusion = nn.Sequential(
            nn.Linear(2048*3, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.embedding = nn.Linear(2048, 512)
        self.classifier = nn.Linear(2048, num_classes)

    def get_features(self, x):
        B, T = x.shape[:2]
        x = x.view(-1, *x.shape[2:])
        frame_features = self.backbone(x)
        _, C, H, W = frame_features.shape
        frame_features = frame_features.view(B, T, C, H, W)
        
        # Global features
        global_features = frame_features.mean(dim=(1,3,4))
        
        # Temporal features
        patches = frame_features.unfold(3, 4, 4).unfold(4, 4, 4)
        patches = patches.contiguous().view(B, T, C, -1, 16)
        patches = patches.mean(dim=-1).permute(0,1,3,2)
        temporal_nodes = patches.reshape(B, -1, C)
        temporal_features = self.temporal_gcn(temporal_nodes).max(dim=1)[0]
        
        # Spatial features
        spatial_features = self.spatial_gcn(patches).mean(dim=[1,2])
        
        return torch.cat([global_features, temporal_features, spatial_features], dim=1)
        
    def forward(self, x):
        features = self.feature_fusion(self.get_features(x))
        return self.classifier(features)

# Sampler
class BalancedPKsampler(torch.utils.data.Sampler):
    def __init__(self, dataset, P=32, K=4):
        self.dataset = dataset
        self.P = min(P, len(dataset.unique_pids))
        self.K = K
        self.length = self.P * self.K
        self.class_counts = {pid: len(indices) for pid, indices in dataset.pid_to_indices.items()}
        self.valid_pids = [pid for pid, count in self.class_counts.items() if count >= K]

    def __iter__(self):
        batch = []
        pids = random.sample(self.valid_pids, self.P)
        for pid in pids:
            indices = self.dataset.pid_to_indices[pid]
            selected = random.sample(indices, self.K) if len(indices) >= self.K else random.choices(indices, k=self.K)
            batch.extend(selected)
        random.shuffle(batch)
        return iter(batch)
    
    def __len__(self):
        return self.length

# Triplet Mining
def hard_triplet_mining(embeddings, labels, margin=0.3):
    pairwise_dist = torch.cdist(embeddings, embeddings)
    anchors, positives, negatives = [], [], []
    
    for i in range(len(embeddings)):
        label = labels[i]
        pos_mask = (labels == label)
        pos_mask[i] = False
        if pos_mask.any():
            hardest_pos = torch.argmax(pairwise_dist[i][pos_mask])
            positives.append(embeddings[hardest_pos])
        else:
            positives.append(embeddings[i])
            
        neg_mask = (labels != label)
        if neg_mask.any():
            hardest_neg = torch.argmin(pairwise_dist[i][neg_mask])
            negatives.append(embeddings[hardest_neg])
        else:
            negatives.append(embeddings[i])
            
        anchors.append(embeddings[i])
    
    return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)

# Evaluation
def compute_metrics(model, gallery_loader, query_loader, device):
    model.eval()
    
    @torch.no_grad()
    def extract_features(loader):
        feats, pids, camids = [], [], []
        for videos, batch_pids, batch_camids in loader:
            videos = videos.to(device, non_blocking=True)
            features = model.feature_fusion(model.get_features(videos))
            feats.append(features.cpu())
            pids.extend(batch_pids.numpy())
            camids.extend(batch_camids.numpy())
        return torch.cat(feats), np.array(pids), np.array(camids)
    
    gallery_feats, gallery_pids, gallery_camids = extract_features(gallery_loader)
    query_feats, query_pids, query_camids = extract_features(query_loader)
    
    query_feats = query_feats.numpy()
    gallery_feats = gallery_feats.numpy()
    dist_mat = cdist(query_feats, gallery_feats, metric='cosine')
    
    def compute_map(dist, query_pids, gallery_pids, query_camids, gallery_camids):
        aps = []
        for i in range(len(query_pids)):
            valid = gallery_camids != query_camids[i]
            matches = (gallery_pids[valid] == query_pids[i]).astype(float)
            if matches.sum() > 0:
                aps.append(average_precision_score(matches, -dist[i, valid]))
        return np.mean(aps) if aps else 0.0
    
    def compute_rank(dist, query_pids, gallery_pids, query_camids, gallery_camids, rank):
        cmc = []
        for i in range(len(query_pids)):
            valid = gallery_camids != query_camids[i]
            matches = (gallery_pids[valid] == query_pids[i]).astype(float)
            if matches.sum() > 0:
                order = np.argsort(dist[i, valid])
                cmc.append(matches[order].cumsum()[rank-1])
        return np.mean(cmc) * 100 if cmc else 0.0
    
    return {
        'mAP': compute_map(dist_mat, query_pids, gallery_pids, query_camids, gallery_camids),
        'rank1': compute_rank(dist_mat, query_pids, gallery_pids, query_camids, gallery_camids, 1),
        'rank5': compute_rank(dist_mat, query_pids, gallery_pids, query_camids, gallery_camids, 5),
        'rank10': compute_rank(dist_mat, query_pids, gallery_pids, query_camids, gallery_camids, 10)
    }

# Training
def train():
    # Initialize datasets
    train_set = MSMT17Dataset(Config.dataset_paths["train"], transform)
    gallery_set = MSMT17Dataset(Config.dataset_paths["test"], transform)
    query_set = MSMT17Dataset(Config.dataset_paths["query"], transform)

    print(f"Train samples: {len(train_set)}")
    print(f"Gallery samples: {len(gallery_set)}")
    print(f"Query samples: {len(query_set)}")

    # Data loaders
    train_loader = DataLoader(
        train_set,
        batch_size=Config.batch_size,
        sampler=BalancedPKsampler(train_set, Config.P, Config.K),
        num_workers=Config.num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    gallery_loader = DataLoader(
        gallery_set, 
        batch_size=Config.batch_size, 
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True
    )
    query_loader = DataLoader(
        query_set, 
        batch_size=Config.batch_size, 
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True
    )

    # Model
    model = STGCN(len(train_set.unique_pids)).to(Config.device)
    
    # Loss and optimizer
    criterion = {
        'cross_entropy': nn.CrossEntropyLoss(),
        'triplet': nn.TripletMarginLoss(margin=Config.margin)
    }
    optimizer = AdamW([
        {'params': model.backbone.parameters(), 'lr': 3e-5},
        {'params': list(model.temporal_gcn.parameters()) + 
                 list(model.spatial_gcn.parameters()), 'lr': 3e-4},
        {'params': list(model.feature_fusion.parameters()) + 
                 list(model.embedding.parameters()) +
                 list(model.classifier.parameters()), 'lr': 3e-4}
    ])
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.num_epochs)
    scaler = GradScaler()
    best_map = 0.0
    
    # Load checkpoint
    checkpoint_path = os.path.join(Config.checkpoint_dir, "last_checkpoint.pth")
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_map = checkpoint['best_map']
            print(f"Resuming from epoch {start_epoch}, best mAP: {best_map:.4f}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}, starting from scratch")
            start_epoch = 0
    else:
        start_epoch = 0

    # Training loop
    for epoch in range(start_epoch, Config.num_epochs):
        model.train()
        epoch_start = time.time()
        
        for videos, pids, _ in train_loader:
            videos = videos.to(Config.device, non_blocking=True)
            pids = pids.to(Config.device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast():
                features = model.feature_fusion(model.get_features(videos))
                embeddings = model.embedding(features)
                logits = model.classifier(features)
                
                anchors, positives, negatives = hard_triplet_mining(embeddings, pids)
                loss = criterion['cross_entropy'](logits, pids) + \
                      0.5 * criterion['triplet'](anchors, positives, negatives)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        
        scheduler.step()
        epoch_time = time.time() - epoch_start
        
        # Evaluation
        if (epoch + 1) % 2 == 0 or epoch == Config.num_epochs - 1:
            metrics = compute_metrics(model, gallery_loader, query_loader, Config.device)
            print(f"Epoch {epoch+1}/{Config.num_epochs} | Time: {epoch_time:.2f}s | "
                  f"mAP: {metrics['mAP']:.4f} | Rank-1: {metrics['rank1']:.2f} | "
                  f"Rank-5: {metrics['rank5']:.2f} | Rank-10: {metrics['rank10']:.2f}")
            
            if metrics['mAP'] > best_map:
                best_map = metrics['mAP']
                torch.save(model.state_dict(), os.path.join(Config.model_dir, "best_model.pth"))
                print(f"New best model saved with mAP: {best_map:.4f}")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_map': best_map,
        }, checkpoint_path)

if __name__ == "__main__":
    train()