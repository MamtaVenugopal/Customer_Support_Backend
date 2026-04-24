# Bearing Anomalous Sound Detection: Attribute-Aware Domain Generalization Plan

## 1. Problem Understanding

### Dataset Structure
The bearing development dataset (`bearing_dev`) has three sections, each representing a different type of domain shift:

| Section | Domain Shift Parameter | Source Domain | Target Domain | Attribute Key | Type |
|---------|----------------------|---------------|---------------|---------------|------|
| **00** | Rotation velocity | Slow speed (vel=0-7) | Fast speed (vel=8-14) | `vel` in filename | Continuous |
| **01** | Microphone location | Fixed placement (e.g., top) | Different placement (e.g., side) | `loc` in filename | Categorical (A, B, C, D) |
| **02** | Factory noise | Ambient noise from source factory | Ambient noise from target factory | `f-n` in filename | Categorical (1-5) |

### Training Data per Section
- **990 clips**: source-domain normal sounds
- **10 clips**: target-domain normal sounds  
- **Total**: 1000 normal training clips per section

### Test Data per Section (in `test/` folder)
- **50 source normal + 50 source anomaly** (domain known)
- **50 target normal + 50 target anomaly** (domain known)
- **200 clips total per section**

## 2. Why Attributes Matter for Domain Generalization

### The Core Challenge
Without using attributes, a model trained on 990 slow-speed (source) clips + 10 fast-speed (target) clips has a **problem**: 
- The 10 target clips are a tiny fraction of training data
- Model learns: "velocity changes → normal" isn't explicitly captured
- Result: At test time, the **target-domain normal sounds get high anomaly scores** (false positives)

### How Attributes Solve This
By extracting and binning velocity/location/noise from filenames, we tell the model:
1. **"This clip is slow-speed, not inherently anomalous"** (condition on attribute)
2. **"These 10 clips show what fast-speed normal sounds like"** (teach domain shift pattern)
3. **"Learn features that are robust to velocity changes"** (bottleneck regularization)

This is the essence of **domain generalization**: learning invariant features rather than domain-specific features.

## 3. Three Architecture Approaches

### Architecture 1: Attribute-Aware Autoencoder (Recommended for velocity/continuous shifts)

**Core Idea**: Condition the reconstruction on the attribute bin

**Data Flow**:
```
Input: log-mel spectrogram (T, 128) + attribute_bin (scalar)
↓
Concat: [flattened_spec(T*128,), attr_embedding(8,)] → (T*128+8,)
↓
Encoder: Dense [640+8 → 512 → 256 → 128 → 64]
↓
Bottleneck: 64-dimensional latent vector z
↓
Decoder: Dense [64 → 128 → 256 → 512 → 640]
↓
Output: reconstructed spec (T*128,)
↓
Anomaly Score: MSE(original_spec, reconstructed_spec)
```

**Why This Works**:
- The attribute embedding in the input tells encoder: "expect velocity=8, don't treat fast-speed as anomaly"
- Bottleneck (64 dims) forces learning velocity-invariant features
- Batch norm during training regularizes against overfitting to specific velocities

**Training**:
- Batch size: 32-64
- Optimizer: Adam (lr=1e-3)
- Loss: MSE(input, recon)
- Epochs: 50-100
- Stratify batches: 80% source, 20% target per batch

**Pros**:
- ✅ Explicit domain awareness
- ✅ Learns velocity-invariant reconstruction
- ✅ Interpretable: attribute → latent mapping

**Cons**:
- ❌ Requires attributes at inference (you have them, but adds complexity)
- ❌ Attribute discretization: binning continuous velocity loses some information
- ❌ Slower training than outlier exposure

---

### Architecture 2: MobileNetV2 Section Classifier (Recommended for fast prototyping)

**Core Idea**: Implicit domain shift learning via section classification (outlier exposure)

**Data Flow**:
```
Input: log-mel spectrogram images (64, 128, 1) or tripled to (64, 128, 3)
↓
MobileNetV2 backbone (pre-trained on ImageNet or trained from scratch)
↓
Global Average Pooling
↓
Dense layer → output 3 logits (for 3 sections)
↓
Softmax: [p_sec00, p_sec01, p_sec02]
↓
Anomaly Score: -log(p_true_section) averaged over all patches
```

**Training Objective**:
- Classification: maximize p_true_section for all normal clips (both source and target)
- Section label derived from filename during training
- Train on: all 1000 normal clips (stratified by section)
- Loss: CrossEntropyLoss(pred_logits, true_section_ID)

**Why This Works for Domain Generalization**:
- Section 00 (velocity) has gradual velocity drift across training clips
- MobileNetV2 implicitly learns: "Section 00 clips sound like this across all velocities"
- When a slow-speed normal test clip comes in → recognized as "section 00 normal"
- When a fast-speed normal test clip comes in → same recognition
- Anomaly: both score high (correctly identified as sec00), anomalies score low

**Training**:
- Batch size: 32 (with 64-frame context windows, 100+ patches per clip)
- Optimizer: Adam (lr=1e-5)
- Epochs: 20-30
- Dropout: 0.5 (critical for robustness)
- No warmup needed (low learning rate)

**Pros**:
- ✅ Fast training (20-30 epochs vs. 50-100 for AE)
- ✅ No attributes needed at inference
- ✅ Compact: single model for all sections/bearing (if needed)
- ✅ Outlier exposure: learns what different operating conditions look like

**Cons**:
- ❌ Implicit domain learning: harder to debug why it works
- ❌ Confuses section with domain shift: if section IDs are poorly separated, fails
- ❌ Memory-intensive: requires image batching

---

### Architecture 3: Normalizing Flow (For extreme domain shifts)

**Core Idea**: Learn density p(x) instead of reconstruction, anomalies are low-density

**Only use if** Architecture 1 and 2 both fail on target domain (AUC_target < 50%).

**Data Flow**:
```
Input: log-mel spectrogram
↓
VAE Encoder → latent z ~ N(0, 1)
↓
Normalizing flow: f(z) → w (bijective transform)
↓
Decoder: x_recon = p(x|w)
↓
Anomaly Score: -log p(x) = -log p(z) - sum(log |det Jacobian|)
```

**Why This Works**:
- Probability-based scoring is more principled than reconstruction error
- NF explicitly models density over multiple velocity/noise conditions
- Anomalies have low density in the learned manifold

**Pros**:
- ✅ Theoretically sound for domain generalization
- ✅ Handles extreme shifts better than reconstruction

**Cons**:
- ❌ Complex to implement correctly
- ❌ Slow training (100+ epochs)
- ❌ Hard to debug; use only as last resort

---

## 4. Recommended Development Path

### Phase 1: Prototype (Week 1)

**Goal**: Understand baseline performance, identify which sections are hard

**Steps**:
1. Implement Architecture 2 (MobileNetV2)
   - Fast to code, quick results
   - Check if it handles each section
2. Compute metrics per section:
   ```
   For each section (00, 01, 02):
     AUC_source = AUC(scores on source test, labels)
     AUC_target = AUC(scores on target test, labels)
     pAUC = AUC at FPR ≤ 0.1
     
     If AUC_source > 75% and AUC_target > 65% → section OK
     If AUC_target < 55% → section needs work
   ```
3. Identify problem sections (likely 01 and 02 due to categorical shift)

### Phase 2: Add Attribute Awareness (Week 2)

**Goal**: Improve failing sections with Architecture 1

**Steps**:
1. For sections with AUC_target < 55%:
   - Switch to Architecture 1 (AE with attributes)
   - Extract and bin attributes from training filenames
   - Concat attribute embedding to input
2. Retrain AE on problematic section
3. Re-evaluate: compare AUC_target before/after

### Phase 3: Ensemble (Week 3, if needed)

**Goal**: Combine strengths of both architectures

**Strategy**:
- Use Architecture 2 (MobileNetV2) scores + Architecture 1 (AE) scores
- Weighted ensemble: `final_score = 0.6 * ae_score + 0.4 * mobilenet_score`
- Optimize weights on dev test split

### Phase 4: Evaluation on Additional Training Data (Week 4)

Once development dataset is tuned:
- Retrain on sections 03, 04, 05 (additional training dataset)
- Same architecture and hyperparameters
- Evaluate on evaluation dataset (sections 03, 04, 05)

---

## 5. Data Pipeline: Implementation Details

### Step 1: Attribute Extraction from Filenames

**Bearing attribute format** (from DCASE docs):
- Section 00: `section_00_source_train_normal_vel_<velocity>_<index>_.wav`
- Section 01: `section_01_source_train_normal_vel_<velocity>_loc_<location>_<index>_.wav`
- Section 02: `section_02_source_train_normal_vel_<velocity>_f-n_<index>_.wav`

**Code pattern**:
```python
import re

def extract_attributes(filename):
    """
    Returns: {
        'section': int,
        'domain': str,  # 'source' or 'target'
        'label': str,   # 'normal' or 'anomaly'
        'vel': int,     # 0-14
        'loc': str,     # A-D (section 01 only)
        'f_n': int,     # 1-5 (section 02 only)
    }
    """
    match = re.match(r'section_(\d+)_(source|target)_train_(normal|anomaly)_(.+?)_(\d+)_.wav', filename)
    if not match:
        return None
    
    section, domain, label, attrs_str, idx = match.groups()
    attrs = {'section': int(section), 'domain': domain, 'label': label}
    
    # Parse variable-length attributes
    attr_parts = attrs_str.split('_')
    i = 0
    while i < len(attr_parts):
        if attr_parts[i] == 'vel':
            attrs['vel'] = int(attr_parts[i+1])
            i += 2
        elif attr_parts[i] == 'loc':
            attrs['loc'] = attr_parts[i+1]
            i += 2
        elif attr_parts[i] == 'f-n' or attr_parts[i] == 'f' and i+1 < len(attr_parts) and attr_parts[i+1] == 'n':
            # Handle "f-n" split by underscore
            attrs['f_n'] = int(attr_parts[i+2])
            i += 3
        else:
            i += 1
    
    return attrs
```

### Step 2: Attribute Binning

**Goal**: Convert continuous velocities to discrete bins for embedding

```python
def bin_attributes(attrs, section):
    """Discretize attributes for embedding"""
    bins = {}
    
    if 'vel' in attrs:
        # Velocity 0-14 → 3 bins (slow, medium, fast)
        vel = attrs['vel']
        bins['vel_bin'] = 0 if vel < 5 else (1 if vel < 10 else 2)
    
    if 'loc' in attrs:
        # Location A, B, C, D → 0, 1, 2, 3
        loc_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        bins['loc_bin'] = loc_map[attrs['loc']]
    
    if 'f_n' in attrs:
        # Noise 1-5 → 0-4 (already categorical)
        bins['noise_bin'] = attrs['f_n'] - 1
    
    return bins
```

### Step 3: Dataset Creation (PyTorch DataLoader)

```python
import torch
from torch.utils.data import Dataset, DataLoader

class BearingDataset(Dataset):
    def __init__(self, file_list, load_spec_fn, section, transform=None):
        """
        file_list: list of (filepath, attributes_dict)
        load_spec_fn: function to load and compute mel-spec
        section: int (0, 1, 2)
        """
        self.files = file_list
        self.load_spec = load_spec_fn
        self.section = section
        self.transform = transform
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        filepath, attrs = self.files[idx]
        
        # Load spectrogram (T, 128)
        spec = self.load_spec(filepath)  # numpy (T, 128)
        
        # Extract attribute bin
        attr_bin = bin_attributes(attrs, self.section)
        
        # For AE (Architecture 1):
        # Return (spec, attr_bin, domain_label)
        # For MobileNetV2 (Architecture 2):
        # Return (spec, section_id, domain_label)
        
        return {
            'spec': torch.FloatTensor(spec),
            'attr_bin': attr_bin,
            'domain': 0 if attrs['domain'] == 'source' else 1,  # 0=src, 1=tgt
            'section': self.section,
        }

# Usage
def get_train_loader(section, batch_size=32, val_split=0.1):
    train_files = load_bearing_files(f'bearing_dev/bearing/train/', section=section)
    
    # Stratify by domain
    source_files = [f for f in train_files if f[1]['domain'] == 'source']
    target_files = [f for f in train_files if f[1]['domain'] == 'target']
    
    # 80% source, 20% target per batch
    train_dataset = BearingDataset(
        source_files[:int(0.9*len(source_files))] + target_files[:int(0.9*len(target_files))],
        load_spec_fn=compute_mel_spec,
        section=section
    )
    
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
```

### Step 4: Training Loop (Architecture 1 Example)

```python
def train_ae_architecture1(section, num_epochs=100):
    train_loader = get_train_loader(section, batch_size=32)
    
    # Model initialization
    input_dim = 640 + 16  # (5 frames * 128) + (attr embedding)
    model = AttributeAwareAE(input_dim=input_dim, bottleneck_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(num_epochs):
        losses = []
        for batch in train_loader:
            specs = batch['spec']  # (B, T, 128)
            attr_bins = batch['attr_bin']  # dict with vel_bin, etc.
            
            # Create context windows dynamically
            windowed_specs = create_context_windows(specs, window_size=5)  # (B*num_windows, 5, 128)
            
            # Embed attributes
            attr_embeddings = embed_attributes(attr_bins)  # (B*num_windows, 16)
            
            # Concat and forward
            input_combined = torch.cat([windowed_specs.flatten(1), attr_embeddings], dim=1)
            recon, latent = model(input_combined)
            
            loss = torch.nn.functional.mse_loss(input_combined[:, :640], recon)
            losses.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {sum(losses)/len(losses):.4f}")
```

### Step 5: Inference & Scoring

```python
def compute_anomaly_scores(model, test_clip_path, attrs, model_type='ae'):
    """
    Compute per-clip anomaly score
    """
    spec = load_and_compute_mel_spec(test_clip_path)  # (T, 128)
    
    if model_type == 'ae':
        # Architecture 1
        attr_embedding = embed_attributes(attrs)  # (16,)
        
        scores = []
        for t in range(spec.shape[0] - 5):
            window = spec[t:t+5]  # (5, 128)
            input_combined = torch.cat([
                torch.FloatTensor(window).flatten(),
                attr_embedding
            ])
            
            recon, _ = model(input_combined.unsqueeze(0))
            window_score = ((input_combined[:640] - recon)**2).mean().item()
            scores.append(window_score)
        
        return np.mean(scores)  # scalar anomaly score
    
    elif model_type == 'mobilenetv2':
        # Architecture 2
        scores = []
        for t in range(spec.shape[0] - 64):
            image = spec[t:t+64]  # (64, 128)
            image_batch = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)  # (1, 1, 64, 128)
            
            # Triple to RGB if needed
            image_batch = image_batch.repeat(1, 3, 1, 1)
            
            with torch.no_grad():
                logits = model(image_batch)  # (1, 3)
                
            # Section ID from attributes (or training label)
            true_section = attrs['section']
            anomaly_score = -torch.log_softmax(logits[0], dim=0)[true_section].item()
            scores.append(anomaly_score)
        
        return np.mean(scores)  # scalar anomaly score
```

### Step 6: Evaluation Metrics

```python
def evaluate_section(model, section, model_type='ae'):
    """
    Compute AUC_source, AUC_target, pAUC for a single section
    """
    test_files = load_bearing_files(f'bearing_dev/bearing/test/', section=section)
    
    source_files = [f for f in test_files if f[1]['domain'] == 'source']
    target_files = [f for f in test_files if f[1]['domain'] == 'target']
    
    # Compute scores
    source_scores = []
    source_labels = []
    for filepath, attrs in source_files:
        score = compute_anomaly_scores(model, filepath, attrs, model_type)
        source_scores.append(score)
        source_labels.append(1 if attrs['label'] == 'anomaly' else 0)
    
    target_scores = []
    target_labels = []
    for filepath, attrs in target_files:
        score = compute_anomaly_scores(model, filepath, attrs, model_type)
        target_scores.append(score)
        target_labels.append(1 if attrs['label'] == 'anomaly' else 0)
    
    # Metrics
    from sklearn.metrics import roc_auc_score
    
    auc_source = roc_auc_score(source_labels, source_scores)
    auc_target = roc_auc_score(target_labels, target_scores)
    
    # pAUC: AUC at FPR ≤ 0.1
    from roc_auc_partial import partial_auc
    auc_mixed = partial_auc(source_labels + target_labels, 
                           source_scores + target_scores, 
                           fpr_max=0.1)
    
    print(f"Section {section:02d}")
    print(f"  AUC (source): {auc_source:.4f}")
    print(f"  AUC (target): {auc_target:.4f}")
    print(f"  pAUC [0, 0.1]: {auc_mixed:.4f}")
    
    return auc_source, auc_target, auc_mixed
```

---

## 6. Success Criteria

| Metric | Target | Interpretation |
|--------|--------|-----------------|
| **AUC (source)** | > 75% | Model can detect anomalies in original domain |
| **AUC (target)** | > 65% | Model generalizes to shifted domain |
| **pAUC [0, 0.1]** | > 55% | Good precision at low false positive rates (practical) |
| **Harmonic mean** | > 65% | Overall domain generalization score |

### Per-Section Expectations

- **Section 00 (velocity shift)**: Likely to be easiest (continuous, gradual)
  - Architecture 2 alone may suffice
  - If not: Architecture 1 with velocity binning
  
- **Section 01 (location shift)**: Harder (categorical, discrete placement)
  - May need Architecture 1 with explicit location bins
  
- **Section 02 (noise shift)**: Hardest (environmental, multimodal)
  - Likely needs Architecture 1 with noise category bins
  - Or ensemble of both architectures

---

## 7. Implementation Checklist

- [ ] **Data loading**
  - [ ] Parse attributes from filenames
  - [ ] Create train/val/test splits stratified by domain
  - [ ] Implement mel-spectrogram computation (STFT 64ms, 128 bands)
  
- [ ] **Architecture 2 (MobileNetV2) - Prototype**
  - [ ] Load pre-trained MobileNetV2
  - [ ] Add classification head (3 sections)
  - [ ] Train on 990 source + 10 target per section
  - [ ] Evaluate: AUC_source, AUC_target, pAUC per section
  - [ ] Identify failing sections
  
- [ ] **Architecture 1 (AE + Attributes) - For difficult sections**
  - [ ] Implement attribute embedding layer (concat to input)
  - [ ] Build encoder-bottleneck-decoder
  - [ ] Train with stratified batching (80% source, 20% target)
  - [ ] Evaluate on same metrics
  - [ ] Compare to Architecture 2
  
- [ ] **Threshold and decision**
  - [ ] Fit gamma distribution to training/dev test anomaly scores
  - [ ] Determine 90th percentile threshold
  - [ ] Generate decision_result_*.csv files
  
- [ ] **Evaluation on development dataset**
  - [ ] Compute AUC and pAUC for all sections
  - [ ] Compute harmonic mean across all sections/domains
  - [ ] Document per-section performance
  
- [ ] **Prepare for additional training and evaluation datasets**
  - [ ] Retrain on sections 03, 04, 05 (same hyperparameters)
  - [ ] Evaluate on sections 03, 04, 05 test set

---

## 8. Key References & Resources

1. **DCASE 2022 Task 2 baseline code**:
   - https://github.com/Kota-Dohi/dcase2022_task2_baseline_ae
   - https://github.com/Kota-Dohi/dcase2022_task2_baseline_mobile_net_v2

2. **Domain generalization techniques**:
   - Domain-mixing: concatenate source + target training
   - Attribute conditioning: explicit feature factorization
   - Outlier exposure: learn discriminative boundaries

3. **Evaluation**:
   - pAUC (partial AUC) at FPR ≤ 0.1 is critical for practical systems
   - Harmonic mean weights all metrics equally (no section has preference)

---

## Summary

This plan provides three complementary architectures for handling domain shifts in bearing anomalous sound detection:

1. **Start simple**: Architecture 2 (MobileNetV2 section classifier) for fast prototyping
2. **If needed, go explicit**: Architecture 1 (AE with attribute conditioning) for fine-grained control
3. **Last resort**: Architecture 3 (normalizing flows) for extreme shifts

The key insight is **leveraging attributes from filenames** to tell the model about domain shifts explicitly, enabling faster convergence and better target-domain performance.

Good luck!
