import argparse
import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def evaluate_predictions(folder_path: str, prompts_csv_path: str, output_dir: str):
    """
    Liest Vorhersage-Dateien aus einem Ordner, berechnet Evaluationsmetriken
    und gibt einen formatierten Bericht sowie eine Konfusionsmatrix aus.
    """
    # --- 1. Daten laden ---
    try:
        files = sorted([f for f in os.listdir(folder_path) if f.endswith('.pt')])
        if not files:
            print(f"Fehler: Kein.pt-Dateien im Ordner '{folder_path}' gefunden.")
            return
    except FileNotFoundError:
        print(f"Fehler: Der angegebene Ordner '{folder_path}' existiert nicht.")
        return

    all_preds = []
    all_labels = []
    for f in tqdm(files, desc="Lade Ergebnisse"):
        try:
            data = torch.load(os.path.join(folder_path, f), map_location=torch.device('cpu'))
            all_preds.append(data['pred'])
            all_labels.append(data['label'])
        except Exception as e:
            print(f"\nWarnung: Konnte Datei {f} nicht laden oder verarbeiten. Fehler: {e}")

    if not all_labels:
        print("Fehler: Konnte keine validen Label-Daten aus den Dateien extrahieren.")
        return

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    # --- 2. Klassennamen laden ---
    try:
        prompts_df = pd.read_csv(prompts_csv_path)
        # Eindeutige Klassennamen in der richtigen Reihenfolge extrahieren
        class_names = prompts_df.sort_values('classidx').drop_duplicates('classidx')['classname'].tolist()
    except FileNotFoundError:
        print(f"Fehler: prompts.csv-Datei nicht unter '{prompts_csv_path}' gefunden.")
        print("Klassennamen können nicht angezeigt werden.")
        class_names = [str(i) for i in sorted(np.unique(y_true))]

    # --- 3. Metriken berechnen ---
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
    cm = confusion_matrix(y_true, y_pred)

    # --- 4. Ergebnisse formatiert ausgeben ---
    print("\n" + "="*60)
    print(" E V A L U A T I O N S B E R I C H T")
    print("="*60)
    print(f" Ergebnis-Ordner: {folder_path}")
    print(f" Anzahl der ausgewerteten Samples: {len(y_true)}")
    print("-" * 60)
    print(f" Gesamtgenauigkeit (Accuracy): {accuracy:.2%}")
    print(f" Mittlere Klassengenauigkeit (Balanced Accuracy): {balanced_accuracy:.2%}")
    print("-" * 60)
    print(" Detaillierter Klassifikationsbericht:")
    print(report)
    print("="*60)

    # --- 5. Konfusionsmatrix visualisieren und speichern ---
    plt.figure(figsize=(max(8, len(class_names)), max(6, len(class_names) * 0.8)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 12})
    plt.ylabel('Wahre Klasse', fontsize=14)
    plt.xlabel('Vorhergesagte Klasse', fontsize=14)
    plt.title('Konfusionsmatrix', fontsize=16)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Speichern der Grafik
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, 'konfusionsmatrix.png')
    plt.savefig(output_path, dpi=300)
    print(f" Konfusionsmatrix wurde gespeichert unter: {output_path}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Wertet gespeicherte Vorhersagen aus und erstellt einen detaillierten Bericht sowie eine Konfusionsmatrix."
    )
    parser.add_argument(
        "folder", 
        type=str, 
        help="Pfad zum Ordner, der die.pt-Ergebnisdateien enthält."
    )
    parser.add_argument(
        "--prompts_csv", 
        type=str, 
        required=True, 
        help="Pfad zur prompts.csv-Datei, um Klassennamen zu laden."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None, 
        help="Optionaler Pfad zum Speichern der Konfusionsmatrix. Standardmäßig wird der Input-Ordner verwendet."
    )
    args = parser.parse_args()

    output_directory = args.output_dir if args.output_dir else args.folder
    
    evaluate_predictions(args.folder, args.prompts_csv, output_directory)


if __name__ == '__main__':
    main()