import numpy as np
import pandas as pd
max_lead = 30
dynamic_table = np.zeros((max_lead + 1, max_lead + 1))
attacker_hash_rate = 0.25

# fill diagonal
dynamic_table[0][0] = 1 - attacker_hash_rate
dynamic_table[max_lead][max_lead] = attacker_hash_rate
for distance in range(1, max_lead + 1):
    # fill transition from state i to state i + distance
    for i in range(max_lead - distance + 1):
        j = i + distance
        if distance == 1:
            dynamic_table[i][j] = attacker_hash_rate
    # fill transition from state i to state i - distance
    for i in range(distance, max_lead + 1):
        j = i - distance
        if distance == 1:
            dynamic_table[i][j] = 1 - attacker_hash_rate
stationary_distribution = np.linalg.matrix_power(dynamic_table, 1000)
df = pd.DataFrame(stationary_distribution)
print(df)

verification = []
C = stationary_distribution[0][0]
fraction = 1/3
for i in range(10):
    verification.append(C * (fraction**i))
print(stationary_distribution[0][:10])
print(verification)