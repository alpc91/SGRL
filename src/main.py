import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tqdm import trange

import utils
from arguments import get_args
from config import *

from subproc_vec_env import SubprocVecEnv

from common.buffer import ReplayBuffer
from common.logger import Logger
from common.util import set_device_and_logger, load_config, set_global_seed, merge_dict

from trainer import Trainer
from agent import Agent

import wandb
import socket
from common import util


class AttrDict(dict):
    def __init__(self,*args,**kwargs):
        super(AttrDict,self).__init__(*args,**kwargs)
        self.__dict__= self


def main():
    args = get_args()
    args = vars(args)
    args = merge_dict(args,load_config(args["config_path"], {}))
    for key in args:
        if type(args[key]) == dict:
            args[key] = AttrDict(args[key])
    args = AttrDict(args)
    print(args)

    #set global seed
    set_global_seed(args.seed)
    logger = Logger(args.exp_path, args.env_name,seed=args.seed, info_str = args.expID, print_to_terminal=True)

    #set device and logger
    set_device_and_logger(args.gpu, logger)

    #save args
    args.wandb_id = wandb.util.generate_id()
    logger.log_str_object("parameters", log_dict = args)

    # Retrieve MuJoCo XML files for training ========================================
    logger.log_str("Initializing Environment")
    args.envs_train_names = []

    args.traversal_types = ['pre', 'inlcrs', 'postlcrs']
    args.graph_dicts = dict()
    args.graphs = dict()
    # existing envs
    if not args.custom_xml:
        for morphology in args.morphologies:
            args.envs_train_names += [
                name[:-4]
                for name in os.listdir(args.xml_path)
                if ".xml" in name and morphology in name
            ]
        for name in args.envs_train_names:
            args.graphs[name] = utils.getGraphStructure(
                os.path.join(args.xml_path, "{}.xml".format(name)),
                args.observation_graph_type,
            )
            args.graph_dicts[name] = utils.getGraphDict(
                    args.graphs[name], args.traversal_types, device=util.device
                )
    # custom envs
    else:
        if os.path.isfile(args.custom_xml):
            assert ".xml" in os.path.basename(args.custom_xml), "No XML file found."
            name = os.path.basename(args.custom_xml)
            env_name = name[:-4]
            args.envs_train_names.append(env_name)  # truncate the .xml suffix
            args.graphs[env_name] = utils.getGraphStructure(
                args.custom_xml, args.observation_graph_type
            )
            args.graph_dicts[env_name] = utils.getGraphDict(
                    args.graphs[env_name], args.traversal_types, device=util.device
                )
        elif os.path.isdir(args.custom_xml):
            for name in os.listdir(args.custom_xml):
                if ".xml" in name:
                    env_name = name[:-4]
                    args.envs_train_names.append(env_name)
                    args.graphs[env_name] = utils.getGraphStructure(
                        os.path.join(args.custom_xml, name), args.observation_graph_type
                    )
                    args.graph_dicts[env_name] = utils.getGraphDict(
                            args.graphs[env_name], args.traversal_types, device=util.device
                        )    


    args.envs_train_names.sort()
    args.num_envs_train = len(args.envs_train_names)
    print("#" * 50 + "\ntraining envs: {}\n".format(args.envs_train_names) + "#" * 50)

    #initialize environment
    args.limb_obs_size,args.limb_action_size, args.max_action, args.observation_space, args.action_space = utils.registerEnvs(
        args.envs_train_names, args.max_episode_steps, args.custom_xml
    )
    print(args.limb_obs_size,args.limb_action_size)
    args.max_num_limbs = max([len(args.graphs[env_name]) for env_name in args.envs_train_names])
    # create vectorized training env
    args.obs_max_len = (
        args.max_num_limbs
        * args.limb_obs_size
    )
    args.action_max_len = (
        args.max_num_limbs
        * args.limb_action_size
    )
    if args.actor_type ==  'mlp':
        args.obs_max_len = args.observation_space[args.envs_train_names[-1]].shape[0]
        args.action_max_len = args.action_space[args.envs_train_names[-1]].shape[0]+3
    print("max_num_limbs,args.obs_max_len,args.action_max_len",args.max_num_limbs,args.obs_max_len,args.action_max_len)
    # determine the maximum number of children in all the training envs
    if args.max_children is None:
        args.max_children = utils.findMaxChildren(args.envs_train_names, args.graphs)
    args.rel_size = args.graph_dicts[args.envs_train_names[0]]['relation'].shape[-1]
       

    envs_train = [
        utils.makeEnvWrapper(name, obs_max_len=args.obs_max_len, seed=args.seed) for name in args.envs_train_names
    ]
    envs_eval = [
        utils.makeEnvWrapper(name, obs_max_len=args.obs_max_len, seed=args.seed) for name in args.envs_train_names
    ]
    # vectorized env
    envs_train = SubprocVecEnv(envs_train)  
    envs_eval = SubprocVecEnv(envs_eval)  

    #initialize buffer
    logger.log_str("Initializing Buffer")
    # different replay buffer for each env; avoid using too much memory if there are too many envs
    env_buffer = dict()
    # model_buffer = dict()
    if args.num_envs_train > args.rb_max // 1e6:
        for name in args.envs_train_names:
            env_buffer[name] = ReplayBuffer(args.observation_space[name], args.action_space[name],
            max_buffer_size = int(args.rb_max // args.num_envs_train), modular=True)
            # model_buffer[name] = ReplayBuffer(args.observation_space[name], args.action_space[name],
            # max_buffer_size = int(args.rb_max // args.num_envs_train),modular=True)
    else:
        for name in args.envs_train_names:
            # print(args.max_action,args.action_space[name],args.action_space[name].shape,args.action_space[name].high,args.action_space[name].low)
            env_buffer[name] = ReplayBuffer(args.observation_space[name], args.action_space[name],
            max_buffer_size = args.env_buffer["max_buffer_size"],modular=True)
            # model_buffer[name] = ReplayBuffer(args.observation_space[name], args.action_space[name],
            # max_buffer_size = args.model_buffer["max_buffer_size"],modular=True)
        
    
    # #initialize agent
    logger.log_str("Initializing Agent")
    agent = Agent(args)

    #initialize trainer
    logger.log_str("Initializing Trainer")
    trainer  = Trainer(
        agent,
        envs_train,
        envs_eval,
        env_buffer,
        args,
        **args['trainer']
    )

    
    wandb.init(
        id = args.wandb_id,
        config=args,
        project=args.env_name, 
        entity="modular_rl",
        notes=socket.gethostname(),
        name=args.expID+"_s"+str(args.seed),
        dir=logger.log_path,
        tags=["modular", "SubprocVecEnv"],
        resume="allow",
        reinit=True
        )
    logger.log_str("Started training")
    trainer.train()
    wandb.finish()



if __name__ == "__main__":
    main()