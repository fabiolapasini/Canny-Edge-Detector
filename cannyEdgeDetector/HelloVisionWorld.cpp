// FABIOLA PASINI	313585	fabiola.pasini@studenti.unipr.it

// non ho usato padding
// es 1) e 2) ho utilizzato le formule nelle slide per trovare la nuova
// dimensione dell'immagine di uscita e scorro le immagini senza uscire dai
// bordi es 3) e 4) divido per 255 per rendere tra 0 e 1 i valori da assegnare
// all'immagine es 4) conferto da CV_32F a CV_8U moltiplicando per 255 e ho
// riportato i valori tra 0 e 255
// nota: per il constrant stretching nell'es 7) ho utilizzato questa formula:
// VAL_MAX ((pixel-min) / (max-min)*255), ma nell'es 4) si semplicava -> min=0 e
// max=255.
/* es 6) decommentare l'es 5) per utilizzarlo
                la media di valori e positiva, quindi ho ignorato il caso
   negativo. nell es prima applico il filtro verticale sull immagine di partenza
   e poi sul risultande del filtro orizontale trovato nell es 5). per trasporre
   il kernel1D ho utilizzato la funzione .t() */
// es 7) su entrambe le matrici ho eseguito il constant stretching tra 0 e i
// rispettivi valori massimi 8) l'esercizio e pensato per un'immagine CV_8U,
// nell'es 9) viene applicato su una CV_32F quindi ho fatto i due casi

#include <iostream>
#include <filesystem>
#include <cstdlib>
#include <opencv2/opencv.hpp>

#include "src/utils.cpp"
#include "include/utils.h"

using namespace cv;
using namespace std;


int main(int argc, char** argv) {
  runCannyEdgeDetector();
  return 0;
}
