# Prozessüberwachung mittels Self-organizing map
Zum Verständnis diese Quellen anschauen:
* [Erklärung SOM Algorithmus](http://www.ai-junkie.com/ann/som/som1.html)
* [Erklärung TensorFlow Code für den Algorithmus](https://codesachin.wordpress.com/2015/11/28/self-organizing-maps-with-googles-tensorflow/)

### Workflow
Der Workflow kann im workflow IPython Notebook nachvollzogen werden und läuft folgendermaßen:

* Die Rohdaten werden mit dem Beschleunigungssensor aufgenommen und in Fertigungszyklen (Samples) zerlegt
* Am einfachsten geth das über einlesen der DB in Pandas und das speichern der Zyklen als .csv
* Die Samples werden mittels RawDataConverter alle gleich eingelesen und die STF-Transformation duchgeführt
* Die Daten werden analog zu sklearn Methodik als X_train, X_test, y_train, y_test returned
* Allerdings geht das nur mit gelabelten Daten
* Ungelabelte Daten können auch über die .csv von einer RawDataConverter-Instanz transformiert werden
* Mittels der so erstellten Daten wird die SOM trainiert
* Über Clustering der Modellvektoren der SOM können nach dem trainieren ungesehene Daten klassifiziert werden
* Die Automaton-Klasse wird mit einer/mehreren Liste(n) an Labels, z.B. von der SOM erzeugt, trainiert
* Anschließend können andere Listen an Labels auf einige Abweichungen die auf Anomalie hinweisen untersucht werden

### Visualisierung
Im examples Notebook befinden sich Beispiele für die Visualisierungsmöglichkeiten der SOM
