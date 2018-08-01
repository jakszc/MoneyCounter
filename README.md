# MoneyCounter
Projekt z Komunikacji Człowiek-Komputer (PUT, Semestr 5).

Skrypt korzystający z różnych technik rozpoznawania obrazu (wykrywanie konturów, analizowanie kolorów, porównywanie -na bardzo podstawowym poziomie- z znanymi obiektami) celem policzenia wartości pieniędzy (monety i banknoty w zakresie 5gr - 20zł) widocznych na zdjęciu wejściowym.

Jedynym warunkiem dla prawidłowego działania algorytmu jest, aby monety/banknoty leżały oddzielnie, jednak najlepsze efekty osiąga się gdy zdjęcie wykonane jest na ciemnym, jednolitym, nieodblaskującym tle, oświetlonym światłem białym.

Aby zmienić analizowane zdjęcie, należy zaktualizować 427 linijkę skryptu na:  
org_img = cv2.imread("./wejscie/NAZWA_OBRAZU.jpg")  
gdzie NAZWA_OBRAZU to interesujące nas zdjęcie w folderze "wejscie".

Wynikiem działania skryptu są wycięte fragmenty obrazu z rozpoznanymi monetami/banknotami oraz oryginalne zdjęcie z nałożoną maską na rozpoznanych kształtach i oszacowaną sumą kwoty z lewym, górnym rogu zdjęcia.
