from servers import Server
from validators import ValidatorClassification, ValidatorConfig
from models import Cifar10CnnModel,Digit5CNN,MnistMLP,LettersCNN
from solvers import ServerSolver, Cifar10CNNClientSolver,MnistClientSolver
from utils import init_exp, logger
from clients_preparation import *
import time

def run_classification_exp(exp_id):
    num_epochs = 200
    model_fn = Digit5CNN
    server_solver = ServerSolver(estimation_interval=2,
                                 do_selection=True,
                                 selection_size=15
                                 )
    client_solver = MnistClientSolver()
    create_data_start_time = time.time()
    client_vec, test_ds_vec, preparation_dict = create_clients_digits5_5set(client_solver)
    print('time for creating data is {}'.format(time.time()-create_data_start_time))
    validator_config = ValidatorConfig(num_class=preparation_dict['num_classes'], num_epochs=num_epochs,
                                       do_cluster_eval=True, do_client_eval=True, do_importance_estimation=True,
                                       client_eval_idx_vec=range(preparation_dict['num_clients']))

    logger.log_server_solver(exp_id, server_solver.to_json_dict())
    logger.log_client_solver(exp_id, client_solver.to_json_dict())
    logger.log_client_preparation(exp_id, preparation_dict)
    logger.log_model_description(exp_id, model_fn)

    server = Server(model_fn=model_fn, client_vec=client_vec, num_clusters=preparation_dict['num_clusters'],
                    server_solver=server_solver, validator=ValidatorClassification(test_ds_vec, validator_config,'digits'),
                    exp_id=exp_id)
    server.run(num_global_epochs=num_epochs)


def main():
    exp_id = 'experiment_3_mnist_4set'
    init_exp(exp_id)
    logger.log_exp_info(exp_id,
                        description='experiment description')
    run_classification_exp(exp_id)


if __name__ == '__main__':
    main()
