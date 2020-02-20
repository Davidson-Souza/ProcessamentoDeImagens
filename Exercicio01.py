import random
#Sorteia o número
sorteado = random.randint(-1, 20)
# Recupera o palpite da pessoa
palpite = int (input("Digite um número: "))

def verifica():
    global palpite, sorteado
    # Verifica se a pessoa acertou
    if palpite == sorteado:
        print("Acertou!")
    else:   # Caso tenha errado...
        # verifica se o palpite é maior ou menor que o valor sorteado...
        if ( palpite > sorteado):
            print("Mais baixo...")
        else:
            print("Mais alto...")
        palpite = int (input("Digite um número: "))
        verifica()
verifica()
