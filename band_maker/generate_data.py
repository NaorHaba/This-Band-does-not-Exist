import argparse
from band_maker.band_generator import BandGenerator
from band_maker.custom_generate import decrease_temperature_gradually
from functools import partial
import numpy as np

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--batch_size", type=int, help="Number of bands to generate.")
    args_parser.add_argument("--save_path", type=str, help="Path to save generated data to.")
    args = args_parser.parse_args()

    bg = BandGenerator('models/gpt2_forward_model', None, 'data/artists_blacklist.pickle', 'data/genres.pickle')
    bg.generate_batch(batch_size=args.batch_size,
                      max_iterations=np.inf,
                      temperature=2.0,
                      transform_logits_warper=partial(decrease_temperature_gradually,
                                                      decrease_factor=0.85),
                      top_k=300,
                      num_return_sequences=12,max_length=1024, save_path=args.save_path)