#!/usr/bin/env python3
"""Wrap an EXISTING dynamic evaluator with filtered contact logging.

Example form, with real installed evaluator/config/checkpoint arguments:
 python t2_run_dynamic_with_contacts.py --entry /actual/evaluator.py \
     --out /fresh/contact-log-directory -- <existing evaluator arguments>

Do not use replay.py: its reference visualization is not a controller rollout.
Caller must configure a single environment and a dynamic object. This wrapper
does not guess evaluator CLI flags, choose a checkpoint, or alter training.
"""
import argparse
from pathlib import Path
import runpy
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entry',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--object-key')
    parser.add_argument('arguments',nargs=argparse.REMAINDER)
    args=parser.parse_args();entry=args.entry.expanduser().resolve()
    if not entry.is_file():parser.error('Evaluator entry point does not exist.')
    if entry.name=='replay.py':parser.error('Use a dynamic evaluator, not the reference replay.')
    from holosoma.utils import eval_utils
    original=eval_utils.init_sim_imports
    def start(config,*a,**kw):
        app=original(config,*a,**kw)
        from isaac_filtered_contact_hooks import install_dynamic_contact_logging
        install_dynamic_contact_logging(args.out,args.object_key)
        return app
    eval_utils.init_sim_imports=start
    remaining=args.arguments[1:] if args.arguments[:1]==['--'] else args.arguments
    sys.argv=[str(entry),*remaining]
    runpy.run_path(str(entry),run_name='__main__')


if __name__=='__main__':main()
