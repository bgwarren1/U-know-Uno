from uknowuno.game_state import new_game
from ml.featurize import feature_dim
for n in [2,3,4]:
    g = new_game(n_players=n, manual_mode=True)
    print(n, feature_dim(g))
# All three lines should print the same number.
