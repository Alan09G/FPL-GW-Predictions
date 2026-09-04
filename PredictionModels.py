# PREDICTION MODELS

def linear_regression(data):
    data = data[data['gw'] > 1]
    train_df = data[data['gw'] <= 30]
    test_df = data[data['gw'] > 30]

    scaler = StandardScaler()

    X_train = train_df.drop(columns=['gw', 'event_points', 'target_next_points'])
    y_train = train_df['target_next_points']

    X_test = test_df.drop(columns=['gw', 'event_points', 'target_next_points'])
    y_test = test_df['target_next_points']

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    lm = LinearRegression()
    lm.fit(X_train_scaled, y_train)

    predictions = lm.predict(X_test_scaled)
    print("MAE:", metrics.mean_absolute_error(y_test, predictions))
    print('MSE:', metrics.mean_squared_error(y_test, predictions))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

    print('R2 Score: ', r2_score(y_test, predictions))

def decision_tree(data):
    data = data[data['gw'] > 1]
    train_df = data[data['gw'] <= 30]
    test_df = data[data['gw'] > 30]

    X_train = train_df.drop(columns=['gw', 'event_points', 'target_next_points'])
    y_train = train_df['target_next_points']

    X_test = test_df.drop(columns=['gw', 'event_points', 'target_next_points'])
    y_test = test_df['target_next_points']

    clf_gini = DecisionTreeRegressor(criterion='squared_error', max_depth=5, random_state=0)
    clf_gini.fit(X_train, y_train)
    y_pred = clf_gini.predict(X_test)

    print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('R2 Score: ', r2_score(y_test, y_pred))

def random_forest(data):
    data = data[data['gw'] > 1]
    train_df = data[data['gw'] <= 30]
    test_df = data[data['gw'] > 30]

    X_train = train_df.drop(columns=['gw', 'event_points', 'target_next_points'])
    y_train = train_df['target_next_points']

    X_test = test_df.drop(columns=['gw', 'event_points', 'target_next_points'])
    y_test = test_df['target_next_points']

    rfc = RandomForestRegressor(n_estimators=500, max_depth=4, random_state=0, n_jobs=-1)
    rfc.fit(X_train, y_train)

    rfc_pred = rfc.predict(X_test)
    print('MAE:', metrics.mean_absolute_error(y_test, rfc_pred))
    print('MSE:', metrics.mean_squared_error(y_test, rfc_pred))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, rfc_pred)))
    print('R2 Score: ', r2_score(y_test, rfc_pred))

def xgboost(data):
    data = data[data['gw'] > 1]
    train_df = data[data['gw'] <= 30]
    test_df = data[data['gw'] > 30]

    X_train = train_df.drop(columns=['gw', 'event_points', 'target_next_points'])
    y_train = train_df['target_next_points']

    X_test = test_df.drop(columns=['gw', 'event_points', 'target_next_points'])
    y_test = test_df['target_next_points']

    model = xgb.XGBRegressor(n_estimators=1000, max_depth=6, learning_rate=0.01, random_state=0)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print('MAE:', metrics.mean_absolute_error(y_test, predictions))
    print('MSE:', metrics.mean_squared_error(y_test, predictions))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
    print('R2 Score: ', r2_score(y_test, predictions))
