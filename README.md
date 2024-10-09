# AutoEssayScorer
Dieses Repository enthält ein neuronales Netz zur automatisierten Aufsatzbewertung, das mit einem LSTM-Modell verglichen wird. Ziel ist es, die Genauigkeit und Effizienz der automatisierten Bewertung zu verbessern.


## Inhaltsverzeichnis

- [Überblick](#überblick)
- [Funktionen](#funktionen)
- [Installation](#installation)
- [Verwendung](#verwendung)
- [Modellvergleich](#modellvergleich)
- [Datensatz](#datensatz)
- [Ergebnisse](#ergebnisse)
- [Kontakt](#kontakt)

## Überblick

Die automatisierte Aufsatzbewertung ist eine Anwendung der natürlichen Sprachverarbeitung (NLP), die darauf abzielt, die Qualität geschriebener Aufsätze durch maschinelle Lernalgorithmen zu bewerten. Dieses Projekt konzentriert sich auf den Aufbau eines neuronalen Netzwerkmodells zur Bewertung von Aufsätzen und den Vergleich seiner Leistung mit einem LSTM-basierten Ansatz.

## Funktionen

- **Neuronales Netzwerkmodell**: Implementierung eines neuronalen Netzwerks zur Aufsatzbewertung.
- **LSTM-Modellvergleich**: Evaluierung und Vergleich des neuronalen Netzwerkmodells mit einem LSTM-Modell.
- **Automatisierte Bewertung**: Das Modell dient der Bewertung von Aufsätzen gemäß den Niveaustufen des CEFR.

## Installation

Um das Projekt lokal einzurichten, folgen Sie diesen Schritten:

1. **Repository klonen**:
    ```bash
    git clone https://github.com/yourusername/AutoEssayScorer.git
    cd AutoEssayScorer
    ```

2. **Erforderliche Pakete installieren**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Datensatz herunterladen**:


## Verwendung

Nachdem die Umgebung eingerichtet ist, können Sie mit dem Training der Modelle beginnen:

1. **Neuronales Netzwerkmodell trainieren**:
    ```bash
    python Beispiel
    ```

2. **LSTM-Modell trainieren**:
    ```bash
    python Beispiel
    ```

3. **Modelle evaluieren**:
    ```bash
    python Beispiel
    ```

4. **Ergebnisse visualisieren**:
    ```bash
    python Beispiel
    ```

## Modellvergleich

Dieses Projekt umfasst einen detaillierten Vergleich zwischen dem neuronalen Netzwerk und den LSTM-Modellen. Der Vergleich konzentriert sich auf:

- **Genauigkeit**: ...
- **Effizienz**: ...

## Datensatz

Die von uns verwendeten Daten sind eine Kombination aus gelabelten Aufsätzen aus den DISKO-, Falko- und Merlin-Korpora und den persönlichen Daten (122 Aufsätze - A1) von [Elias](https://github.com/EliasAhlers).

| Dataset                     | A1         | A2         | B1         | B2         | C1         | C2         |
|-----------------------------|------------|------------|------------|------------|------------|------------|
| **Merlin**                   | 57 texts   | 306 texts  | 331 texts  | 293 texts  | 42 texts   | 4 texts    |
| **Falko_C2**                 | 0 texts    | 0 texts    | 0 texts    | 0 texts    | 0 texts    | 95 texts   |
| **Falko_Mixed**              | 0 texts    | 0 texts    | 0 texts    | 87 texts   | 90 texts   | 70 texts   |
| **Elias_A1**                 | 122 texts  | 0 texts    | 0 texts    | 0 texts    | 0 texts    | 0 texts    |
| **DISKO**                    | 0 texts    | 0 texts    | 27 texts   | 216 texts  | 294 texts  | 58 texts   |
|                              |            |            |            |            |            |            |
| **Our Data**                 |            |            |            |            |            |            |
| **All_Data_with_features**   | 179 texts  | 306 texts  | 358 texts  | 596 texts  | 426 texts  | 227 texts  |
| **All_Data_with_features_new** | 179 texts  | 306 texts  | 358 texts  | 552 texts  | 426 texts  | 227 texts  |

**All_Data_with_features_new** : Total 2048 Texts (Optimized for the model: -44 Texts from B2)


Der in diesem Projekt verwendete Datensatz enthält Aufsätze in deutscher Sprache mit verschiedenen Merkmalen und Sprachniveaus. Die Sprachniveaus für die deutsche Sprache in Aufsätzen sind in die Kategorien `NCefr` und `Cefr` unterteilt. Innerhalb des NCefr wird diese Kategorie mit drei Stufen (A, B, C) bewertet, während im CEFR eine Einteilung in sechs Stufen (A1, A2, B1, B2, C1, C2) erfolgt.

### Merkmale des Datensatzes:

- **NCefr & Cefr**: Sprachniveaus der Aufsätze
- **Text**: Aufsätze
- **Grammar Errors**: Anzahl der Grammatikfehler im Aufsatz, ermittelt durch das Tool `LanguageTool`.
- **Type Token Ratio**: Verhältnis der einzigartigen Wörter (Types) zu den gesamten Wörtern (Tokens) im Text.
- **Lexical Density**: Lexikalische Dichte, die das Verhältnis von Inhaltswörtern zu Funktionswörtern misst.
- **Flesch Reading Ease**: Lesbarkeitsindex nach Flesch, angepasst an die deutsche Sprache.
- **Wiener Sachtextformel Average**: Ein weiterer Lesbarkeitsindex, spezifisch für deutsche Texte.

Diese Merkmale werden verwendet, um die Qualität, Komplexität und Lesbarkeit der Essays zu bewerten.


## Ergebnisse

Das Projekt liefert Ergebnisse basierend auf:

- **Modellgenauigkeit**: 
- **Trainingszeit**: 
- **Ressourcennutzung**: 


## Kontakt

Bei Fragen oder für Unterstützung kontaktieren Sie bitte [Nurhayat Altunok](mailto:nualt100@uni-duesseldorf.de).
