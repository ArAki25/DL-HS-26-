# Zusammenarbeit

## Ablauf

1. Aktuellen Stand holen: `git checkout main && git pull`
2. Eigenen Branch anlegen: `git checkout -b <kürzel>/<thema>` (z. B. `jhonny/data-exploration`)
3. Arbeiten und oft committen: `git add <dateien>` und `git commit -m "Kurze Beschreibung"`
4. Pushen: `git push -u origin <branch>`
5. Auf GitHub einen Pull Request öffnen, eine Person aus dem Team schaut drüber, dann mergen.

## Regeln

- Nie direkt auf `main` pushen.
- Keine Daten, Modelle (`.pt`, `.pth`) und keine `.env` committen. Das `.gitignore` schützt davor, trotzdem vor dem Commit mit `git status` prüfen.
- Notebooks vor dem Commit mit **Restart & Run All** laufen lassen, damit sie reproduzierbar sind. Pro Notebook arbeitet nur eine Person gleichzeitig, sonst gibt es Merge-Konflikte.
- Neue Pakete in `requirements.txt` eintragen und im PR erwähnen.
- Gemeinsam genutzter Code kommt nach `src/`, nicht in Notebooks kopieren.
- Wer KI-Hilfe (z. B. Claude) nutzt, versteht und kann erklären, was der Code macht. Die Deklarationspflicht für KI-Nutzung im Modul beachten.
