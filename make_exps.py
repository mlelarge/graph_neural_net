from commander import ex

noise_list = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]

extended_noise_list = [0.005 * i for i in range(0,41)]

repeat = 10

print("Running ErdosRenyi experiments")
for noise in noise_list:
    for _ in range(repeat):
        ex.run(config_updates={
            'name': 'ErdosRenyi',
            'cpu': False,
            'data': {'noise': noise},
            })

print("Running Regular experiments")
for noise in noise_list:
    for repeat in range(repeat):
        ex.run(config_updates={
            'name': 'Regular',
            'cpu': False,
            'data': {
                'generative_model': 'Regular',
                'noise': noise},
            })

print("Running extended experiments")
for noise in extended_noise_list:
    for repeat in range(repeat):
        ex.run(config_updates={
            'name': 'Regular',
            'cpu': False,
            'data': {
                'generative_model': 'Regular',
                'noise': noise,
                'vertex_probab': 0.9},
            })
