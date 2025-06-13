def neurofeedback(score, **p): print("[NF]", score)
def game(score, **p):          print("[GAME]", score)

OUTPUT_FUNCS = {"Neurofeedback": neurofeedback, "Game": game}
