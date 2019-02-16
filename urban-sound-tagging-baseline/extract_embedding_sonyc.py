import argparse
import gzip
import os
import numpy as np
from tqdm import tqdm

import keras
import tensorflow as tf
from kapre.time_frequency import Melspectrogram
from .vggish import vggish_input
from .vggish import vggish_postprocess
from .vggish import vggish_slim

from ..models import MODELS
from .extract_embedding import extract_embedding as extract_sonyc_embedding
from .sonyc_data import load_sonyc_data

TARGET_DURATION = 1.0  # seconds


def make_extract_vggish_embedding(frame_duration, hop_duration, input_op_name='vggish/input_features',
                                  output_op_name='vggish/embedding', embedding_size=128, resources_dir=None):
    """
    Creates a coroutine generator for extracting and saving VGGish embeddings

    Parameters
    ----------
    frame_duration
    hop_duration
    input_op_name
    output_op_name
    embedding_size
    resources_dir

    Returns
    -------
    coroutine

    """
    params = {
        'frame_win_sec': frame_duration,
        'frame_hop_sec': hop_duration,
        'embedding_size': embedding_size
    }

    if not resources_dir:
        resources_dir = os.path.join(os.path.dirname(__file__), '../../resources/vggish')

    pca_params_path = os.path.join(resources_dir, 'vggish_pca_params.npz')
    model_path = os.path.join(resources_dir, 'vggish_model.ckpt')

    try:
        with tf.Graph().as_default(), tf.Session() as sess:
            # Define the model in inference mode, load the checkpoint, and
            # locate input and output tensors.
            vggish_slim.define_vggish_slim(training=False, **params)
            vggish_slim.load_vggish_slim_checkpoint(sess, model_path, **params)

            while True:
                # We use a coroutine to more easily keep open the Tensorflow contexts
                # without having to constantly reload the model
                audio_path, output_path = (yield)

                examples_batch = vggish_input.wavfile_to_examples(audio_path, **params)

                # Prepare a postprocessor to munge the model embeddings.
                pproc = vggish_postprocess.Postprocessor(pca_params_path, **params)

                input_tensor_name = input_op_name + ':0'
                output_tensor_name = output_op_name + ':0'

                features_tensor = sess.graph.get_tensor_by_name(input_tensor_name)
                embedding_tensor = sess.graph.get_tensor_by_name(output_tensor_name)

                # Run inference and postprocessing.
                [embedding_batch] = sess.run([embedding_tensor],
                                             feed_dict={features_tensor: examples_batch})

                emb = pproc.postprocess(embedding_batch, **params).astype(np.float32)

                with gzip.open(output_path, 'wb') as f:
                    emb.dump(f)

    except GeneratorExit:
        pass


def extract_embeddings_sonyc(annotation_path, audio_dir, output_dir, exp_id, sr, emb_type, model_path=None,
                             model_input_type=None, subnetwork_model_name=None, frame_duration=1.0, hop_duration=0.1,
                             progress=True, vggish_resource_dir='/home/jtc440/dev/l3embedding/resources/vggish',
                             vggish_embedding_size=128):
    """
    Extract embeddings for files annotated in the SONYC annotation file and save them to disk.

    Parameters
    ----------
    annotation_path
    audio_dir
    output_dir
    exp_id
    sr
    emb_type
    model_path
    model_input_type
    subnetwork_model_name
    frame_duration
    hop_duration
    progress
    vggish_resource_dir
    vggish_embedding_size

    Returns
    -------

    """

    file_list = load_sonyc_data(annotation_path)[0]

    frame_length = int(frame_duration * sr)
    hop_length = int(hop_duration * sr)

    if emb_type == 'vggish':
        assert sr == 16000

        extract_vggish_embedding = make_extract_vggish_embedding(frame_duration, hop_duration,
            input_op_name='vggish/input_features', output_op_name='vggish/embedding',
            resources_dir=vggish_resource_dir, embedding_size=vggish_embedding_size)
        # Start coroutine
        next(extract_vggish_embedding)

    elif emb_type == 'sonyc':
        # load model and extract embedding subnet
        assert model_path is not None
        assert subnetwork_model_name is not None
        assert sr == 48000

        target_length = int(np.ceil(sr * TARGET_DURATION))

        model = keras.models.load_model(
            model_path, custom_objects={'Melspectrogram': Melspectrogram})
        emb_model, _, _, _ = MODELS[subnetwork_model_name]()
        emb_model.set_weights(model.get_layer('resnet50').get_weights())

    else:
        raise ValueError('Invalid embedding type: {}'.format(emb_type))

    out_dir = os.path.join(output_dir, exp_id, emb_type)
    os.makedirs(out_dir, exist_ok=True)

    if progress:
        file_list = tqdm(file_list)

    for filename in file_list:
        audio_path = os.path.join(audio_dir, filename)
        emb_path = os.path.join(out_dir, os.path.splitext(filename)[0] + '.npy.gz')

        if emb_type == 'sonyc':
            extract_sonyc_embedding(audio_path, emb_path, emb_model, model_input_type, sr=sr,
                                    frame_length=frame_length, hop_length=hop_length, target_length=target_length)
        elif emb_type == 'vggish':
            extract_vggish_embedding.send((audio_path, emb_path))

    if emb_type == 'vggish':
        extract_vggish_embedding.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("annpath")
    parser.add_argument("audiodir")
    parser.add_argument("embtype")

    parser.add_argument("outputfolder")
    parser.add_argument("expid")

    parser.add_argument("--modelpath")
    parser.add_argument("--modelinputtype")
    parser.add_argument("--subnetwork_model_name")
    parser.add_argument("--vggish_resource_dir")
    parser.add_argument("--vggish_embedding_size", type=int, default=128)

    parser.add_argument("--sr", type=int, default=48000)
    parser.add_argument("--frame_duration", type=float, default=1.0)
    parser.add_argument("--hop_duration", type=float, default=0.1)
    parser.add_argument("--progress", action="store_const", const=True, default=False)

    args = parser.parse_args()

    extract_embeddings_sonyc(annotation_path=args.annpath,
                             audio_dir=args.audiodir,
                             emb_type=args.embtype,
                             output_dir=args.outputfolder,
                             model_path=args.modelpath,
                             model_input_type=args.modelinputtype,
                             exp_id=args.expid,
                             subnetwork_model_name=args.subnetwork_model_name,
                             vggish_resource_dir=args.vggish_resource_dir,
                             vggish_embedding_size=args.vggish_embedding_size,
                             sr=args.sr,
                             frame_duration=args.frame_duration,
                             hop_duration=args.hop_duration,
                             progress=args.progress)
