import numpy as np
import src.scalers as sc


def test_scalers():

    # Local variables
    scaler_name = 'ExpMinMaxScaler'
    want_infinities = True
    n_samples = 10
    features_range = np.array([
        [-1., 10.],
        [70., 100.],
        [-103., 0.],
    ])
    n_features = len(features_range)

    # Generate and rescale data
    data = np.random.rand(n_samples, n_features)
    a = features_range[:, 0]
    b = features_range[:, 1]
    data = (b-a)*data + a

    # In case add infinity
    if want_infinities:
        inf = np.random.randint(
            0,
            high=2,
            size=(n_samples, n_features)).astype(np.float64)
        inf[inf == 1] = np.inf
        data = data + inf

    # Call scaler and rescale
    scaler = sc.Scaler.choose_one(scaler_name)
    scaler.fit(data)
    data_scaled = scaler.transform(data)
    data_inv_scaled = scaler.inverse_transform(data_scaled)

    # Data with infinities replaced
    data_no_inf = scaler._replace_inf(data)

    abs_diff = np.abs(data_inv_scaled/data_no_inf - 1.)

    print('Differences per element:')
    print(abs_diff)
    print()
    print('Minimum differences per feature: {}'.format(abs_diff.min(axis=0)))
    print('Maximum differences per feature: {}'.format(abs_diff.max(axis=0)))

    return
