'''
Lab 6
'''


########## Part 1 ###########

'''
    1)  from sklearn.datasets import load_boston
    Extract the description of all the features and print it
    Split your data into train(80% of data) and test(20% of data) via random selection      
'''
from sklearn.datasets import load_boston
X, y = load_boston(return_X_y=True)
df = load_boston()
print(df.feature_names)
# print("Data:\n",X)
# print("Target:\n",y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


'''
    2)  Try LinearRegression from sklearn.linear_model   
        Try it with and without normalization. Compare the results and pick the best trained model(for comparisson try different metrics from sklearn.metrics like: r2, mse, mae)
        (Hint: for normalizing your data set normalize=True)
    
'''
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


X2_train, X2_test, y2_train, y2_test = train_test_split(X_train, y_train, test_size=0.2)

lr = LinearRegression()
lr.fit(X2_train, y2_train)
lr_pred = lr.predict(X2_test)
print("\nNOT NORMALIZED (LINEAR REGRESSION):")
print("R2 SCORE: ", r2_score(y2_test, lr_pred))
print("MEAN SDQUARED ERROR: ", mean_squared_error(y2_test, lr_pred))
print("MEAN ABSOLUTE ERROR: ", mean_absolute_error(y2_test, lr_pred))


lr_normalized = LinearRegression(normalize=True)
lr_normalized.fit(X2_train, y2_train)
lr_normalized_pred = lr_normalized.predict(X2_test)
print("\nNORMALIZED (LINEAR REGRESSION):")
print("R2 SCORE: ", r2_score(y2_test, lr_normalized_pred))
print("MEAN SDQUARED ERROR: ", mean_squared_error(y2_test, lr_normalized_pred))
print("MEAN ABSOLUTE ERROR: ", mean_absolute_error(y2_test, lr_normalized_pred))

'''
    3) Write the equation of the resulted hyper-plane in Q2.
'''
print()

y_intercept = lr.intercept_
func = "y = "
for i in range(len(lr.coef_)):
    func += f"({lr.coef_[i]:.3f})x{i} + "

func += f"{y_intercept:.3f}"

print(func)

'''
    4)  Repeat Q2 with KNeighborsRegressor. Tune the hyper-parameters(e.g. n_neighbors & metric) using cv techniques. 
'''
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score

for i in range(1,12,2):
    neigh = KNeighborsRegressor(n_neighbors=i, algorithm='auto')
    neigh.fit(X2_train, y2_train)
    neigh_pred = neigh.predict(X2_test)
    print(f"\nNOT NORMALIZED (KNEIGHBOR REGRESSION, k = {i}, algorithm = auto):")
    r2_score_auto = r2_score(y2_test, neigh_pred)
    print("R2 SCORE: ", r2_score_auto)
    print("MEAN SDQUARED ERROR: ", mean_squared_error(y2_test, neigh_pred))
    print("MEAN ABSOLUTE ERROR: ", mean_absolute_error(y2_test, neigh_pred))
    cv_scores = cross_val_score(neigh, X2_train, y2_train, cv=5)
    # print("CROSS VALIDATION SCORES: ", cv_scores)
    print(f"Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
    

    neigh2 = KNeighborsRegressor(n_neighbors=i, algorithm='ball_tree')
    neigh2.fit(X2_train, y2_train)
    neigh_pred2 = neigh2.predict(X2_test)
    print(f"\nNOT NORMALIZED (KNEIGHBOR REGRESSION, k = {i}, algorithm = ball_tree):")
    r2_score_ball_tree = r2_score(y2_test, neigh_pred2)
    print("R2 SCORE: ", r2_score_ball_tree)
    print("MEAN SDQUARED ERROR: ", mean_squared_error(y2_test, neigh_pred2))
    print("MEAN ABSOLUTE ERROR: ", mean_absolute_error(y2_test, neigh_pred2))

    neigh3 = KNeighborsRegressor(n_neighbors=i, algorithm='kd_tree')
    neigh3.fit(X2_train, y2_train)
    neigh_pred3 = neigh3.predict(X2_test)
    print(f"\nNOT NORMALIZED (KNEIGHBOR REGRESSION, k = {i}, algorithm = kd_tree):")
    r2_score_kd_tree = r2_score(y2_test, neigh_pred)
    print("R2 SCORE: ", r2_score_kd_tree)
    print("MEAN SDQUARED ERROR: ", mean_squared_error(y2_test, neigh_pred3))
    print("MEAN ABSOLUTE ERROR: ", mean_absolute_error(y2_test, neigh_pred3))

    neigh4 = KNeighborsRegressor(n_neighbors=i, algorithm='brute')
    neigh4.fit(X2_train, y2_train)
    neigh_pred4 = neigh4.predict(X2_test)
    print(f"\nNOT NORMALIZED (KNEIGHBOR REGRESSION, k = {i}, algorithm = brute):")
    r2_score_brute = r2_score(y2_test, neigh_pred4)
    print("R2 SCORE: ", r2_score_brute)
    print("MEAN SDQUARED ERROR: ", mean_squared_error(y2_test, neigh_pred4))
    print("MEAN ABSOLUTE ERROR: ", mean_absolute_error(y2_test, neigh_pred4))
    


'''
    5) Repeat Q2 with DecisionTreeRegressor from sklearn.tree. Tune the hyper-parameters (e.g. criterion) using cv techniques.
    
'''
from sklearn.tree import DecisionTreeRegressor

dt_regressor = DecisionTreeRegressor()

'''
    6) Which model performs better on the test data?
    
'''

########## Part 2 ###########

'''
    1)  Repeat part 1 with Normalized data. (Hint: use standarscalar from sklearn)
'''
# YOUR CODE GOES HERE  