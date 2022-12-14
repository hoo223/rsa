"""Render trained agents."""
import dl
import argparse
from rsa.ur10_actors import UR10JoystickActor
from rsa.ppo import ConstrainedResidualPPO
from rsa.lunar_lander import LunarLanderJoystickActor
#from rsa.drone_sim import DroneJoystickActor, joystick_agent
#from rsa.ur10.joystick_agent import UR10JoystickActor
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Script.')
    parser.add_argument('logdir', type=str, help='logdir')
    parser.add_argument('--drone', action='store_true', help='conrol drone env')
    parser.add_argument('--reacher', action='store_true', help='conrol luanr reacher env')
    parser.add_argument('--ur10', action='store_true', help='conrol luanr reacher env')
    args = parser.parse_args()

    # if args.drone:
    #     dl.load_config(os.path.join(args.logdir, 'config.gin'),
    #                    ['make_env.env_id="DroneReacherBot-v0"'])
    #     trainer = ConstrainedResidualPPO(args.logdir, nenv=1,
    #                                      base_actor_cls=DroneJoystickActor)
    if args.reacher:
        dl.load_config(os.path.join(args.logdir, 'config.gin'),
                       ['make_env.env_id="LunarLanderReacher-v2"'])
        trainer = ConstrainedResidualPPO(args.logdir, nenv=1,
                                         base_actor_cls=LunarLanderJoystickActor)
    elif args.ur10:
        print("ur10 joystick control")
        dl.load_config(os.path.join(args.logdir, 'config.gin'),
                       ['make_env.env_id="ur10_env:ur10-v0"'])
        trainer = ConstrainedResidualPPO(args.logdir, nenv=1,
                                         base_actor_cls=UR10JoystickActor)
    else:
        dl.load_config(os.path.join(args.logdir, 'config.gin'),
                       ['make_env.env_id="LunarLanderRandomContinuous-v2"'])
        trainer = ConstrainedResidualPPO(args.logdir, nenv=1,
                                         base_actor_cls=LunarLanderJoystickActor)
    trainer.load()
    trainer.evaluate()
    trainer.close()
