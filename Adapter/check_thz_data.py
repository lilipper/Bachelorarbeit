import os

dateipfad = '/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/thz_dataset'
anzahl_dateien = 0
for wurzel, ordner, dateien in os.walk(dateipfad):
    anzahl_dateien += len(dateien)
    for datei in dateien:
        dateipfad = os.path.join(wurzel, datei)

        groesse_in_bytes = os.path.getsize(dateipfad)

        print(f"Dateigröße {datei}: {groesse_in_bytes} Bytes")

print(f"Anzahl der Dateien im Verzeichnis: {anzahl_dateien}")
