Kacper Grzesik 52684
Dominik Górski 52679

# 1. Wstęp i cel ćwiczenia

Celem laboratorium jest zapoznanie się z biblioteką Keras (TensorFlow) oraz praktyczne zastosowanie konwolucyjnych sieci neuronowych do klasyfikacji obrazów medycznych (rozpoznawanie guzów mózgu). Zadanie polegało na załadowaniu danych, zaprojektowaniu trzech architektur sieci neuronowych, przeprowadzeniu eksperymentów oraz ocenie jakości wyników przy użyciu zaawansowanych metryk.

# 2. Przygotowanie i analiza danych

Zbiór danych został wczytany z podziałem na treningowy i testowy. Wszystkie obrazy zostały przeskalowane do 256x256 pikseli. Ze względu na nierówną ilość obrazów w zbiorze danych, zastosowano metrykę Recall zamiast Accuracy (kod sprawdza nierówność dynamicznie)

![[Pasted image 20260116114135.png]]

![[Pasted image 20260116114300.png]]

![[Pasted image 20260116115022.png]]
# Architektury

Zastosowano 3 typy architektur:

### Shallow
Jedna warstwa konwolucyjna, pooling, flatten

![[Pasted image 20260116114456.png]]

![[Pasted image 20260116114514.png]]


### Standard
Trzy bloki konwolucyjne o rosnącej liczbie filtrów

![[Pasted image 20260116114544.png]]

![[Pasted image 20260116114600.png]]

### Deep
GAP (Global Average Pooling)

![[Pasted image 20260116114710.png]]

![[Pasted image 20260116114731.png]]

# Ewaluacja najlepszego modelu

Dla najlepszej architektury, shallow model, przeprowadzono szczegółową analizę na zbiorze testowym.

![[Pasted image 20260116114833.png]]

Precision (Weighted): 0.8065
F1-Score (Weighted):  0.8065
BAC (Balanced Acc.):  0.7961

# Podsumowanie

Eksperyment wykazał, że w przypadku małych zbiorów medycznych, prostsze architektury często osiągają lepsze wyniki niż głębokie sieci. Zastosowany mechanizm pozwolił na adekwatną ocenę modelu w obliczu niezbalansowania klas.