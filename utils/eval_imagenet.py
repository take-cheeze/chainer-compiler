# Note: This file is a modification from chainercv example.
# https://github.com/chainer/chainercv/blob/master/examples/classification/eval_imagenet.py

import argparse
import os

import numpy as np

import chainer
import chainer.functions as F
from chainer import iterators

from chainercv.datasets import directory_parsing_label_names
from chainercv.datasets import DirectoryParsingLabelDataset
from chainercv.links import FeaturePredictor
from chainercv.links import ResNet101
from chainercv.links import ResNet152
from chainercv.links import ResNet50
from chainercv.links import SEResNet101
from chainercv.links import SEResNet152
from chainercv.links import SEResNet50
from chainercv.links import SEResNeXt101
from chainercv.links import SEResNeXt50
from chainercv.links import VGG16

from chainercv.utils import apply_to_iterator
from chainercv.utils import ProgressHook

import chainer_compiler
from chainer_compiler.utils import input_rewriter


models = {
    # model: (class, dataset -> pretrained_model, default batchsize,
    #         crop, resnet_arch)
    'vgg16': (VGG16, {}, 32, 'center', None),
    'resnet50': (ResNet50, {}, 32, 'center', 'fb'),
    'resnet101': (ResNet101, {}, 32, 'center', 'fb'),
    'resnet152': (ResNet152, {}, 32, 'center', 'fb'),
    'se-resnet50': (SEResNet50, {}, 32, 'center', None),
    'se-resnet101': (SEResNet101, {}, 32, 'center', None),
    'se-resnet152': (SEResNet152, {}, 32, 'center', None),
    'se-resnext50': (SEResNeXt50, {}, 32, 'center', None),
    'se-resnext101': (SEResNeXt101, {}, 32, 'center', None),
}


def setup(dataset, model, pretrained_model, batchsize, val, crop, resnet_arch):
    dataset_name = dataset
    if dataset_name == 'imagenet':
        dataset = DirectoryParsingLabelDataset(val)
        label_names = directory_parsing_label_names(val)

    def eval_(out_values, rest_values):
        pred_probs, = out_values
        gt_labels, = rest_values

        accuracy = F.accuracy(
            np.array(list(pred_probs)), np.array(list(gt_labels))).data
        print()
        print('Top 1 Error {}'.format(1. - accuracy))

    if models[model]:
        cls, pretrained_models, default_batchsize = models[model][:3]
        if pretrained_model is None:
            pretrained_model = pretrained_models.get(dataset_name, dataset_name)
    else:
        cls, pretrained_models, default_batchsize = 

    if crop is None:
        crop = models[model][3]
    kwargs = {
        'n_class': len(label_names),
        'pretrained_model': pretrained_model,
    }
    if model in ['resnet50', 'resnet101', 'resnet152']:
        if resnet_arch is None:
            resnet_arch = models[model][4]
        kwargs.update({'arch': resnet_arch})
    extractor = cls(**kwargs)
    model = FeaturePredictor(
        extractor, crop_size=224, scale_size=256, crop=crop)

    if batchsize is None:
        batchsize = default_batchsize

    return dataset, eval_, model, batchsize


def main():
    parser = argparse.ArgumentParser(
        description='Evaluating convnet from ILSVRC2012 dataset')
    parser.add_argument('val', help='Path to root of the validation dataset')
    parser.add_argument('--model', choices=sorted(models.keys()))
    parser.add_argument('--pretrained-model')
    parser.add_argument('--dataset', choices=('imagenet',))
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--batchsize', type=int)
    parser.add_argument('--crop', choices=('center', '10'))
    parser.add_argument('--resnet-arch')
    parser.add_argument('--iterations', '-I', type=int, default=None,
                        help='Number of iterations to train')
    parser.add_argument('--no_use_fixed_batch_dataset',
                        dest='use_fixed_batch_dataset',
                        action='store_false',
                        help='Disable the use of FixedBatchDataset')
    parser.add_argument('--compiler-log', action='store_true',
                        help='Enables compile-time logging')
    parser.add_argument('--trace', action='store_true',
                        help='Enables runtime tracing')
    parser.add_argument('--verbose', action='store_true',
                        help='Enables runtime verbose log')
    parser.add_argument('--skip_runtime_type_check', action='store_true',
                        help='Skip runtime type check')
    parser.add_argument('--dump_memory_usage', action='store_true',
                        help='Dump memory usage')
    parser.add_argument('--quiet_period', type=int, default=0,
                        help='Quiet period after runtime report')
    parser.add_argument('--overwrite_batchsize', action='store_true',
                        help='Overwrite batch size')
    args = parser.parse_args()

    dataset, eval_, model, batchsize = setup(
        args.dataset, args.model, args.pretrained_model, args.batchsize,
        args.val, args.crop, args.resnet_arch)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    if args.export is not None:
        chainer_compiler.use_unified_memory_allocator()
        extractor.to_device(device)
        x = extractor.xp.zeros((args.batchsize, 3, 224, 224)).astype('f')
        chainer_compiler.export(extractor, [x], args.export)
        return

    if args.compile is not None:
        print('run compiled model')
        chainer_compiler.use_chainerx_shared_allocator()
        extractor.to_device(device)
        # init params
        with chainer.using_config('enable_backprop', False),\
                chainer.using_config('train', False):
            x = extractor.xp.zeros((1, 3, 224, 224)).astype('f')
            extractor(x)

        compiler_kwargs = {}
        if args.compiler_log:
            compiler_kwargs['compiler_log'] = True
        runtime_kwargs = {}
        if args.trace:
            runtime_kwargs['trace'] = True
        if args.verbose:
            runtime_kwargs['verbose'] = True
        if args.skip_runtime_type_check:
            runtime_kwargs['check_types'] = False
        if args.dump_memory_usage:
            runtime_kwargs['dump_memory_usage'] = True
            free, total = cupy.cuda.runtime.memGetInfo()
            used = total - free
            runtime_kwargs['base_memory_usage'] = used

        onnx_filename = args.compile
        if args.overwrite_batchsize:
            new_onnx_filename = ('/tmp/overwrite_batchsize_' +
                                 os.path.basename(onnx_filename))
            new_input_types = [
                input_rewriter.Type(shape=(args.batchsize, 3, 224, 224))
            ]
            input_rewriter.rewrite_onnx_file(onnx_filename,
                                             new_onnx_filename,
                                             new_input_types)
            onnx_filename = new_onnx_filename

        extractor_cc = chainer_compiler.compile_onnx(
            extractor,
            onnx_filename,
            'onnx_chainer',
            computation_order=args.computation_order,
            compiler_kwargs=compiler_kwargs,
            runtime_kwargs=runtime_kwargs,
            quiet_period=args.quiet_period)
        model = Classifier(extractor_cc)
    else:
        print('run vanilla chainer model')
        model = Classifier(extractor)

    iterator = iterators.MultiprocessIterator(
        dataset, batchsize, repeat=False, shuffle=False,
        n_processes=6, shared_mem=300000000)

    print('Model has been prepared. Evaluation starts.')
    in_values, out_values, rest_values = apply_to_iterator(
        model.predict, iterator, hook=ProgressHook(len(dataset)))
    del in_values

    eval_(out_values, rest_values)


if __name__ == '__main__':
    main()
