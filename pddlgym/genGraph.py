from PySimpleAutomata import DFA, automata_IO

#dfa_example = automata_IO.dfa_json_importer("DFA_blocks4Colors.json")
#dfa_example = automata_IO.dfa_dot_importer("DFA-Hanoi_4_4.dot")
dfa_example = automata_IO.dfa_json_importer("Full_DFA_Sokoban_6_6.json")

new_dfa=DFA.dfa_reachable(dfa_example)
# new_dfa=DFA.dfa_completion(dfa_example)
# print(new_dfa)
# exit()
# engine='neato'
automata_IO.dfa_to_dot(new_dfa, 'Full_DFA_Sokoban_6_6', './', engine='neato')


# dfa_completion(dfa)