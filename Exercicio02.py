import random

def pedraPapelTesoura():
    maquina = random.randint(1, 3)
    jogador = int (input("Selecione uma opção:\n1 - Pedra\n2 - Papel\n3 - Tesoura\n"))
    
    if ( (maquina == 2) and (jogador == 1) ):
        print("Perdeu! Pedra x Papel")
    elif( maquina == 3 and jogador == 1):
        print("Ganhou! Pedra x Tesoura")
    elif(maquina == 1 and jogador == 1):
        print ("Impate!")
    elif(maquina == 2 and jogador == 2):
        print ("Impate!")
    elif(maquina == 3 and jogador == 3):
        print ("Impate!")
    elif ( maquina == 2 and jogador == 1):
        print("Perdeu! Pedra x Papel")
    elif ( maquina == 2 and jogador == 3):
        print("Ganhou! Tesoura x Papel")
    elif ( maquina == 3 and jogador == 1):
        print("Ganhou! Pedra x Tesoura")
    elif ( maquina == 3 and jogador == 2):
        print("Perdeu! Papel x Tesoura")
pedraPapelTesoura()
