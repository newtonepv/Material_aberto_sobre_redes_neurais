/*mapas explicativos: 

https://lucid.app/lucidspark/cbd1e0e0-43a7-4810-8b9a-b9566c6a480b/edit?beaconFlowId=BB01CA562E07CBE3&invitationId=inv_059155b8-4550-40bb-9cdd-997c624acf34&page=0_0#

https://lucid.app/lucidspark/1f8b320c-8114-4349-af78-05ec1cfd20be/edit?invitationId=inv_42e603f7-68e5-4d2d-86a3-e8d3326ae8b9&page=0_0#
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

double modulo(double x) { //função que devolve o numero em módulo
  if (x < 0) {
    return -x;
  } else {
    return x;
  }
}

double sigmoid(double x) { return 1 / (1 - exp(-x)); } // funcao sigmoid (analogo a ReLU)
double criaPesos() { return (((double)rand()) / ((double)RAND_MAX)) - 0.5; }// funcao que cria os pesos

//input significa entrada
#define numInputs 2 
#define numNos 2
#define numSaidas 1

#define numSetsDeTreinamento 4
#define numPopulacao 10

int main(void) {

  int indexMnErro;
  // declaracoes
  double erroDoIndv[numPopulacao][numSaidas];// guarda o erro do individuo
  double resultadoNos[numPopulacao][numNos]; //guarda os nos, o que esta dentro de cada no
  double biasNos[numPopulacao][numNos]; //guarda o bias de cada no
  double pesoNos[numPopulacao][numInputs][numNos]; //guarda os pesos dos nos da camada do meio
  double biasSaidas[numPopulacao][numSaidas];//guarda o bias da camada de saída
  double pesoSaida[numPopulacao][numSaidas][numNos];//armazena os pesos da camda de saída
  double resultadoSaidas[numPopulacao][numSaidas];//guarda as saidas
  // objetivo da rede neural
  double entradasTreinamento[numSetsDeTreinamento][numInputs] = {
      {0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f}};// entradas
  double saidasTreinamento[numSetsDeTreinamento][numSaidas] = {
      {0.0f}, {1.0f}, {1.0f}, {0.0f}};

  // inicializa

  for (int h = 0; h < numPopulacao; h++) {
    for (int i = 0; i < numNos; i++) {
      for (int j = 0; j < numInputs; j++) {
        pesoNos[h][i][j] = criaPesos();
      }
    }
  }

  for (int h = 0; h < numPopulacao; h++) {
    for (int i = 0; i < numSaidas; i++) {
      for (int j = 0; j < numNos; j++) {
        pesoSaida[h][i][j] = criaPesos();
      }
    }
  }
  // bias
  for (int h = 0; h < numPopulacao; h++) {
    for (int i = 0; i < numSaidas; i++) {
      biasSaidas[h][i] = criaPesos();
    }
  }
  for (int h = 0; h < numPopulacao; h++) {
    for (int i = 0; i < numNos; i++) {
      biasNos[h][i] = criaPesos();
    }
  }
  for (int h = 0; h < numPopulacao; h++) {
    for (int saida = 0; saida < numSaidas; saida++) {
      erroDoIndv[h][saida] = 0;
    }
  }

  int geracoes = 0;
  while (1 + 1 == 2) {
    geracoes++;
    printf("geração %d\n", geracoes);
    for (int h = 0; h < numPopulacao; h++) {
      for (int saida = 0; saida < numSaidas; saida++) {
        erroDoIndv[h][saida] = 0;
      }
    }//isso vai zerar o erro para que nao printemos o erro acumuladamente
    //////////////CALCULO DO ERRO
    
    for (int contIndv = 0; contIndv < numPopulacao; contIndv++) {
      for (int contTreinamento = 0; contTreinamento < numSetsDeTreinamento;
           contTreinamento++) {
        
        for (int contSaida = 0; contSaida < numSaidas; contSaida++) {
          double somatorioSaid = biasSaidas[contIndv][contSaida];

          for (int contNos = 0; contNos < numNos; contNos++) {
            double somatorioNo = biasNos[contIndv][contNos];

            for (int contInp = 0; contInp < numInputs; contInp++) {
              somatorioNo += (entradasTreinamento[contTreinamento][contInp] *
                              pesoNos[contIndv][contNos][contInp]);
            }
            resultadoNos[contIndv][contNos] = sigmoid(somatorioNo);
            somatorioSaid += resultadoNos[contIndv][contNos];
          }
          resultadoSaidas[contIndv][contSaida] = sigmoid(somatorioSaid);

          erroDoIndv[contIndv][contSaida] +=
              modulo((saidasTreinamento[contTreinamento][contSaida] -
                      resultadoSaidas[contIndv][contSaida]));
        }
      }
      printf("erro do individuo %d = %.4f ", contIndv + 1,
             erroDoIndv[contIndv][0]);
      if ((contIndv + 1) % 2 == 0) {
        printf("\n");
      }
    }

    // encontra o menor erro
    double menorErro = erroDoIndv[2][0];//pegamos um qualquer para comparar com o resto na hora de achar o menor

    for (int indv = 0; indv < numPopulacao; indv++) {//percorremos denovo a lista de individuos (pq eu fiz errado, dava pra ter botado no while que calcula o erro de uma vez so)
      if (erroDoIndv[indv][0] < menorErro) {
        menorErro = erroDoIndv[indv][0];
        indexMnErro = indv;
      }
    }
    printf("erro do melhor (%d) = %.4f\n", indexMnErro+1,
       erroDoIndv[indexMnErro][0]);
    
    if (erroDoIndv[indexMnErro][0] < 0.1) {
    break;
    }
    
    // atualizaPesos (mutamos o melhor individuo e copiamos ele em todos)
    
    for (int indiv = 0; indiv < numPopulacao; indiv++) {
      if (indiv != indexMnErro) {

        for (int contNo = 0; contNo < numNos; contNo++) {
          biasNos[indiv][contNo] = biasNos[indexMnErro][contNo] +
          (((double)rand() / (double)RAND_MAX) / 50)-0.01;

          for (int contInput = 0; contInput < numInputs; contInput++) {
            pesoNos[indiv][contInput][contNo]=
            pesoNos[indexMnErro][contInput][contNo]+
          ((((double)rand() / (double)RAND_MAX) / 50) - 0.01);
            
          }
        }
        
          for (int contS = 0; contS < numSaidas; contS++) {
            
            biasSaidas[indiv][contS] =
                biasSaidas[indexMnErro][contS] +
                (((double)rand() / (double)RAND_MAX) / 50) - 0.01;
            
            for (int contNo = 0; contNo < numNos; contNo++) {
              
            pesoSaida[indiv][contS][contNo] =
                pesoSaida[indexMnErro][contS][contNo] +
                (((double)rand() / (double)RAND_MAX) / 50) - 0.01;
          }
        }
      }
    }
    
    
    printf("\n");
  }
  return 0;
}