# README – Model rozpoznawania języka migowego (PJM/ASL)

## Użyte  narzędzia
Python + Mediapipe
Dodatkowe biblioteki: tensorflow, opencv-python, scikit-learn, numpy, pandas, pyarrow, tqdm, kaggle

---

## Typowy przepływ pracy

```
1. Przygotuj dane → dataset_pjm/nazwa_znaku/*.npy
2. Wytrenuj model → python train_lstm.py --classes "znak1,znak2,znak3"
3. Nagraj lub pobierz film z miganiem
4. Uruchom predykcję → python predict_signs.py --source "film.mp4"
```

---

## Wymagania systemowe
- Python >3.11
- Windows 10/11
- Brak wymagań dotyczących GPU — działa na CPU

---

## Struktura plików

```
folder_projektu/
│
├── README.md                   - plik z instrukcją (ten, który obecnie czytasz)
├── requirements.txt            - lista bibliotek wymagana do instalacji
├── load_asl_google.py          - skrypt przygotowywujący dane treningowe
├── train_lstm.py               - skrypt trenujący model
├── predict_signs.py            - skrypt do predykcji na filmie
├── pjm_lstm_model.keras        - wytrenowany model (generowany przez trening)
├── label_map_pjm.json          - mapa klas (generowany przez trening)
├── hand_landmarker.task        - model detekcji dłoni (pobierany automatycznie)
│
├── asl_signs/                  - folder z nieprzygotowanymi danymi treningowymi
│   ├── 2044/
│   │   ├── 635217.parquet
│   │   ├── 3127189.parquet
│   │   └── ...
│   ├── 4718/
│   │   └── ...
│   └── ...
└── dataset_pjm/                - folder z przygotowanymi danymi treningowymi
    ├── apple/
    │   ├── seq_001.npy
    │   ├── seq_002.npy
    │   └── ...
    ├── dog/
    │   └── ...
    └── ...
```

Każdy plik `.npy` to jedna sekwencja — **30 klatek × 63 cechy** (21 punktów dłoni × x, y, z).

---

## Środowisko – jednorazowa konfiguracja

```powershell
py -3.11 -m venv sign_env
.\sign_env\Scripts\Activate.ps1
pip install -r requirements.txt
```

Środowisko trzeba aktywować **za każdym razem** po otwarciu nowego okna PowerShell:
```powershell
.\sign_env\Scripts\Activate.ps1
```

---

## 1. Pobranie znaków do nauki modelu - `load_asl_google.py`

Obecnie nie da się już przystąpić do konkursu, więc folder `asl_signs` został załączony do repozytorium. Jest w nim jedynie część znaków dostępnych z konkursu.
Przy obecnym już folderze `asl_signs` wystarczy uruchomić skrypt:
```powershell
python load_asl_google.py
```

Poniżej dawna instrukcja:
1. Założyć konto na https://www.kaggle.com/account/login
2. Wejść na stronę konkursu i zaakceptować zasady poprzez kliknięcie "Join the competition" https://www.kaggle.com/competitions/sign-language-recognition/data
3. Pobrać dane do nauki modelu poniższym skryptem
```powershell
kaggle auth login
kaggle competitions download -c asl-signs
unzip asl-signs.zip -d asl_signs/
python load_asl_google.py
```

## 2. Skrypt trenujący – `train_lstm.py`

### Jak działa

1. Wczytuje sekwencje `.npy` z folderu `dataset_pjm/`
2. Dzieli dane na trening / walidację / test
3. Trenuje model LSTM z 5-krotną walidacją krzyżową (K-Fold)
4. Zapisuje gotowy model do `pjm_lstm_model.keras`
5. Zapisuje mapę klas do `label_map_pjm.json`
6. Generuje wykresy do folderu `plots/`

Przykładowe wykresy dostępne w folderze `examples`.

### Komendy

**Trenuj na wszystkich słowach w datasecie:**
```powershell
python train_lstm.py
```

**Trenuj tylko na konkretnych słowach:**
```powershell
python train_lstm.py --classes "apple,dog,fish,cry,dance"
```

**Trenuj na pierwszych N słowach (alfabetycznie):**
```powershell
python train_lstm.py --num-classes 10
```

**Zmień liczbę epok i batch size:**
```powershell
python train_lstm.py --epochs 50 --batch 64
```

**Trenuj z konkretnych klas i ustaw własny folder z danymi:**
```powershell
python train_lstm.py --data "C:\moje_dane" --classes "pies,kot,ryba"
```

**Wymuś przeładowanie danych (ignoruj cache):**
```powershell
python train_lstm.py --refresh-cache
```

**Wyłącz cache całkowicie:**
```powershell
python train_lstm.py --no-cache
```

### Wszystkie parametry treningu

| Parametr | Domyślnie | Opis |
|---|---|---|
| `--data` | `dataset_pjm` | Folder z danymi `.npy` |
| `--classes` | `all` | Lista słów po przecinku, np. `"apple,dog,fish"` |
| `--num-classes` | `0` (wszystkie) | Ile klas wziąć (pierwszych N alfabetycznie) |
| `--epochs` | `10` | Maksymalna liczba epok |
| `--batch` | `32` | Rozmiar batcha |
| `--cache` | włączony | Używaj cache dla szybszego ładowania |
| `--no-cache` | — | Wyłącz cache |
| `--refresh-cache` | — | Przeładuj dane i odśwież cache |

### Wyniki treningu

Po zakończeniu w folderze `plots/` pojawią się:
- `training_history.png` — wykres straty i dokładności
- `confusion_matrix.png` — macierz pomyłek
- `confusion_matrix_normalized.png` — znormalizowana macierz pomyłek

---

## 3. Skrypt predykcji – `predict_signs.py`

### Jak działa

1. Przy pierwszym uruchomieniu automatycznie pobiera model detekcji dłoni (~25 MB)
2. Otwiera wskazany plik wideo (można sięgnąć do pliku `fish.mp4` dostępnego w folderze `examples`)
3. Na każdej klatce wykrywa dłoń i wyciąga 63 cechy (punkty dłoni)
4. Buforuje 30 klatek, po czym uruchamia model LSTM
5. Wyświetla przewidywany znak na ekranie
6. Film zapętla się automatycznie

### Komendy

**Podstawowe użycie:**
```powershell
python predict_signs.py --source "C:\ścieżka\do\film.mp4"
```

**Zmień próg pewności (domyślnie 0.5 = 50%):**
```powershell
python predict_signs.py --source "C:\ścieżka\do\film.mp4" --threshold 0.7
```

**Zapisz wynikowy film z nałożoną predykcją:**
```powershell
python predict_signs.py --source "C:\ścieżka\do\film.mp4" --save "wynik.mp4"
```

**Wszystkie opcje naraz:**
```powershell
python predict_signs.py --source "C:\ścieżka\do\film.mp4" --threshold 0.6 --save "wynik.mp4"
```

### Wszystkie parametry predykcji

| Parametr | Domyślnie | Opis |
|---|---|---|
| `--source` | *(wymagany)* | Ścieżka do pliku wideo |
| `--threshold` | `0.5` | Minimalny poziom pewności - poniżej napis jest żółty, powyżej zielony |
| `--save` | brak | Jeśli podany, zapisuje film z predykcją do pliku `.mp4` |

### Co widać na ekranie

- **Górny pasek** - przewidywany znak (zielony = pewny, żółty = niepewny)
- **Pasek pewności** - procent pewności modelu
- **Dolny pasek** - wypełnienie bufora (30 klatek potrzebnych do predykcji) i FPS
- **Zielony szkielet** - wykryte punkty dłoni

### Sterowanie

| Klawisz | Akcja |
|---|---|
| `Q` | Zamknij okno |