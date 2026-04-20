# Klasifikacija melanoma

## Pregled

Ovaj repozitorijum sadrži kompletan **end-to-end pipeline dubokog učenja**
za **binarnu klasifikaciju melanoma iz dermoskopskih snimaka** na
**ISIC 2020** skupu podataka, realizovan u okviru predmeta **Sistemi za
istraživanje i analizu podataka (SIAP)** i **Neuronske mreže** na Fakultetu
tehničkih nauka u Novom Sadu.

**Autori:** Jelena Adamović, Miloš Bojanić

Projekat uključuje:

- **Prilagođenu konvolucionu neuronsku mrežu (Custom CNN)** - baseline sa 3 konvoluciona sloja
- **Transfer learning** sa **EfficientNet B0** arhitekturom
- **Kasnu fuziju (late fusion)** vizuelnih obeležja sa kliničkim metapodacima (pol, starost, anatomska lokacija lezije)
- **Preprocesiranje slika** - uklanjanje dlaka (Black-Hat morfologija + Telea inpainting), resize, normalizacija
- **Ekstrakciju obeležja** - Hu momenti, one-hot enkodiranje metapodataka
- **Offline keširanje** preprocesiranje slika u `.npy` format
- **5-fold StratifiedGroupKFold** unakrsnu validaciju po pacijentu (sprečava curenje podataka)
- **Fokalni gubitak (Focal Loss)** za ekstremnu neuravnoteženost klasa (1,76% malignih)
- **Evaluaciju** - AUC-ROC, Recall, Precision, F1, Youden-ov optimalni prag
- **Analizu pravednosti (fairness)** - Equalized Odds po polu, starosnim grupama i tipu kože
- **Čuvanje modela, rezultata i grafika** u `.pkl` / `.pth` formatu za reprodukciju

Projekat je organizovan za reprodukciju eksperimenata u Google Colab
okruženju (GPU), sa modularnim Python izvornim kodom u `melanoma_colab/src/`.

---

## 📦 Struktura repozitorijuma

    ├── eda.ipynb                                          # Eksplorativna analiza podataka (1. kontrolna tačka)
    ├── Izvestaj_Klasifikacija_melanoma...R2_12-2025.pdf   # Završni izveštaj (SIAP)
    ├── Klasifikacija melanoma...revidirano R2_12-2025.pdf # Revidirana verzija specifikacije za SIAP
    │
    ├── melanoma_colab/                                    # Glavni pipeline za Google Colab
    │   ├── colab_unified.ipynb                            # Glavni notebook: setup + trening + evaluacija + fairness
    │   ├── colab_unified_new.ipynb                        # Proširena verzija sa dodatnim eksperimentima
    │   ├── colab_unified_old.ipynb                        # Starija verzija (arhiva)
    │   ├── colab_setup.ipynb                              # Setup okruženja (Kaggle, Drive)
    │   ├── colab_training_cnn.ipynb                       # Custom CNN trening
    │   ├── colab_training_efficientnet.ipynb              # EfficientNet B0 trening
    │   ├── colab_evaluation.ipynb                         # Evaluacija (ROC, metrike)
    │   ├── colab_fairness.ipynb                           # Fairness analiza
    │   │
    │   └── src/                                           # Python moduli
    │       ├── config.py                                  # Config dataclass (quick_test / small_run / full_run / colab)
    │       ├── preprocessing.py                           # Uklanjanje dlaka, resize
    │       ├── preprocessing_cache.py                     # Offline keširanje u .npy
    │       ├── features.py                                # Hu momenti, encoding metapodataka
    │       ├── augmentation.py                            # Albumentations transformi
    │       ├── dataset.py                                 # MelanomaDataset / CachedMelanomaDataset
    │       ├── data_utils.py                              # Učitavanje CSV-a, StratifiedGroupKFold
    │       ├── models.py                                  # MelanomaCNN + EfficientNetFusion (late fusion)
    │       ├── training.py                                # Train loop, k-fold CV orkestracija
    │       ├── evaluation.py                              # Metrike, matrica konfuzije, optimalni prag
    │       ├── fairness.py                                # Equalized Odds
    │       └── visualization.py                           # Loss krive, ROC, confusion, fairness plots
    │
    └── melanoma_results/                                  # Rezultati treniranja
        ├── cnn_results.pkl                                # Custom CNN: OOF predikcije, metrike, per-fold rezultati
        ├── effnet_results.pkl                             # EfficientNet B0: OOF predikcije, metrike
        └── models/                                        # Sačuvani checkpointi po foldu
            ├── cnn_fold0..4.pth                           # Custom CNN (5 foldova)
            └── efficientnet_fold0..4.pth                  # EfficientNet B0 (5 foldova)

---

# ⚙️ Zahtevi

- **Python 3.10+**
- **PyTorch** i **torchvision**
- **timm** (EfficientNet B0 backbone)
- **albumentations** (augmentacija slika)
- **mahotas** (tekstura, Haralick)
- **OpenCV**, **Pillow** (procesiranje slika)
- **scikit-learn** (StratifiedGroupKFold, metrike)
- **pandas**, **numpy**, **matplotlib**, **seaborn**, **tqdm**

Instalacija svih zavisnosti:

``` bash
pip install torch torchvision timm albumentations mahotas \
            opencv-python Pillow scikit-learn \
            pandas numpy matplotlib seaborn tqdm
```

> Napomena: U Google Colab okruženju, većina zavisnosti je već
> instalirana. Glavni notebook `colab_unified.ipynb` instalira samo
> nedostajuće pakete (`timm`, `albumentations`, `mahotas`).

---

# 📁 Struktura skupa podataka

Projekat koristi **ISIC 2020 Challenge Dataset** sa Kaggle-a:

🔗 [ISIC Challenge Dataset 2020 (Kaggle)](https://www.kaggle.com/datasets/sumaiyabinteshahid/isic-challenge-dataset-2020)

- **33.126** dermoskopskih snimaka
- **2.056** pacijenata
- **Metapodaci:** pol, starost, anatomska lokacija lezije
- **Klase:** benigno (98,24%) i maligno (1,76%)

Očekivana struktura nakon Kaggle download-a (automatski se detektuje u
`colab_unified.ipynb`):

    isic-challenge-dataset-2020/
        ISIC_2020_Training_JPEG/
            train/
                ISIC_0000000.jpg
                ISIC_0000001.jpg
                ...
        ISIC_2020_Training_GroundTruth.csv

---

# 🚀 Kako pokrenuti projekat

Preporučeni način pokretanja je preko **Google Colab-a sa GPU-om** jer
puno treniranje na 33.126 slika zahteva grafičku karticu.

### 1. Eksplorativna analiza podataka (EDA)

**Fajl:** `eda.ipynb`

Ovaj notebook sadrži prvu kontrolnu tačku projekta - istraživačku
analizu ISIC 2020 skupa podataka:

- Distribucija klasa (benigno vs maligno)
- Distribucija metapodataka (pol, starost, anatomska lokacija)
- Statistika snimaka po pacijentima
- Primeri dermoskopskih snimaka
- Detekcija nedostajućih vrednosti

### **Pokretanje**

Lokalno ili u Colab-u:

``` bash
jupyter notebook eda.ipynb
```

### 2. Glavni pipeline - treniranje i evaluacija

**Fajl:** `melanoma_colab/colab_unified.ipynb`

Ovaj notebook pokreće **kompletan pipeline** u jednoj sesiji:

- Setup okruženja (GPU provera, Google Drive mount, Kaggle API)
- Download ISIC 2020 skupa (~23 GB) sa Kaggle-a
- Offline preprocessing i keširanje u `.npy` format
- Treniranje **Custom CNN** (5-fold StratifiedGroupKFold)
- Treniranje **EfficientNet B0** (5-fold StratifiedGroupKFold)
- Evaluacija - ROC krive, confusion matrice, metrike na optimalnom pragu
- Fairness analiza - Equalized Odds po polu i starosnim grupama
- Čuvanje rezultata (`.pkl`) i checkpointa (`.pth`) na Google Drive

### **Priprema pre pokretanja**

1.  Registrovati Kaggle nalog i preuzeti `kaggle.json` API ključ
    (*Settings -> API -> Create New Token*).
2.  Prihvatiti uslove korišćenja ISIC 2020 skupa na Kaggle-u.
3.  Upload-ovati `melanoma_colab/` folder na Google Drive u:

        My Drive/
        └── melanoma_colab/
            ├── src/
            └── colab_unified.ipynb

### **Pokretanje**

1.  Otvoriti `colab_unified.ipynb` u Google Colab-u.
2.  Uključiti GPU: **Runtime -> Change runtime type -> GPU (T4)**.
3.  Pokretati ćelije redom (**Shift + Enter**).
4.  Upload-ovati `kaggle.json` kada se zatraži.

---

### 3. Pokretanje pojedinačnih faza

Ukoliko želiš samo deo pipeline-a, dostupni su i pojedinačni notebook-ovi
u `melanoma_colab/`:

- `colab_setup.ipynb` - samo setup okruženja i Kaggle download
- `colab_training_cnn.ipynb` - samo Custom CNN trening
- `colab_training_efficientnet.ipynb` - samo EfficientNet B0 trening
- `colab_evaluation.ipynb` - samo evaluacija (učitava `.pkl` iz `melanoma_results/`)
- `colab_fairness.ipynb` - samo fairness analiza

---

### 4. Reprodukcija iz koda (Python)

Za programatsku reprodukciju rezultata bez notebook-a:

``` python
from src.config import Config
from src.data_utils import load_and_prepare_data
from src.training import run_cross_validation

cfg = Config.colab()                  # ili Config.full_run()
cfg.model_type = "efficientnet"       # "cnn" ili "efficientnet"
cfg.num_folds = 5
cfg.epochs = 20

df, meta_cols = load_and_prepare_data(cfg)
results = run_cross_validation(df, cfg, metadata_columns=meta_cols)

print(f"Mean AUC: {results['mean_auc']:.4f} +/- {results['std_auc']:.4f}")
```

### **Presetovi konfiguracije**

| Preset | Slike | Epohe | Foldovi | Namena |
|---|---|---|---|---|
| `Config.quick_test()` | 50 | 2 | 2 | Smoke test |
| `Config.small_run()` | 1.482 | 5 | 3 | Lokalni razvoj |
| `Config.full_run()` | 33.126 | 20 | 5 | Puno treniranje (GPU) |
| `Config.colab()` | 33.126 | 20 | 5 | Colab sa Kaggle setupom |

---

### 5. Učitavanje sačuvanog modela

Checkpointi iz `melanoma_results/models/` se učitavaju na sledeći način:

``` python
import torch
from src.config import Config
from src.models import create_model

cfg = Config.colab()
cfg.model_type = "efficientnet"
model = create_model(cfg, metadata_dim=15)
model.load_state_dict(
    torch.load("melanoma_results/models/efficientnet_fold2.pth",
               map_location="cpu")
)
model.eval()
```

---

