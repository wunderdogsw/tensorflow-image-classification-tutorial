# How-to: Image classification with TensorFlow, using Inception v3 neural network

This is a tutorial on how to do image classification using TensorFlow and Inception v3 neural network.
The tutorial is based on ready-made scripts and instructions by TensorFlow authors. The purpose of this tutorial is to put the whole pipeline together in a more understandable way.
Basic knowledge of Python, machine learning, and TensorFlow is a plus.

Note: these instructions won't go much in depth to how everything works.
It's up to you to do the research if you care to know what everything means and happens under the hood.

After you have successfully done image classification, you should be able to start customizing the ready-made scripts etc. for your own image classification needs.

# Installations

Note: This tutorial was created and tested on default MacBook Pro (Retina, 15-inch, Mid 2015), running macOS Sierra

## Clone TensorFlow Serving

    git clone --recurse-submodules https://github.com/tensorflow/serving

TensorFlow itself is included as a submodule.
Another submodule is Tensorflow Models. We will be using "Tensorflow Slim" in this tutorial, which is included in the submodule.

## Install Virtualenv, TensorFlow, Bazel
Instructions:

"Installing from source": https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/setup.md

TensorFlow: https://www.tensorflow.org/install/
Note: If you are using Mac OS X, please use virtualenv, it makes life easier.
Virtualenv: https://virtualenv.pypa.io/en/stable/

Bazel: https://docs.bazel.build/versions/master/install.html

Run TensorFlow configure script and build bazel bins in tensorflow_serving/example/

    $ cd serving/tensorflow
    :/serving/tensorflow $ ./configure
    :/serving $ cd ..
    :/serving $ bazel build -c opt tensorflow_serving/example/...

Building Bazel bins will take a long time.

# Training Inception v3 neural network with test dataset
Note: At this point the instructions assume you have everything installed correctly. If you are running Mac OS X, make sure you are in a Virtualenv.

Inception v3 is a deep convolutional neural network for image classification trained with ImageNet dataset which has 1000 classes.
Read more:
https://www.tensorflow.org/tutorials/image_recognition
https://arxiv.org/abs/1512.00567

We will be fine-tuning the inception v3 to properly classify images with five different classes.
The dataset we are using consists of five flowers. Each flower has hundreds of photos as training data.

First, let's go to the slim directory, which contains necessary python scripts

    cd <path>/<to>/serving/tf_models/slim

Let's add some environment variables in order to make inputting directories a bit easier.
The example uses /tmp/ directory. Modify the variables as you see fit.

    DATA_DIR=/tmp/data/flowers
    CHECKPOINT_DIR=/tmp/checkpoints
    TRAIN_DIR=/tmp/flowers-models/inception_v3

## Download flowers dataset

Download and convert the flowers dataset with a python script that came with Slim

    python download_and_convert_data.py \
        --dataset_name=flowers \
        --dataset_dir="${DATA_DIR}"

Check that DATA_DIR actually contains stuff

    $ ls ${DATA_DIR}
    flowers_train-00000-of-00005.tfrecord
    ...
    flowers_train-00004-of-00005.tfrecord
    flowers_validation-00000-of-00005.tfrecord
    ...
    flowers_validation-00004-of-00005.tfrecord
    labels.txt

## Download Inception v3

Download Inception v3 (assuming you have wget installed)

    CHECKPOINT_DIR=/tmp/checkpoints
    mkdir ${CHECKPOINT_DIR}
    wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
    tar -xvf inception_v3_2016_08_28.tar.gz
    mv inception_v3.ckpt ${CHECKPOINT_DIR}
    rm inception_v3_2016_08_28.tar.gz

## Fine-tune Inception v3

Run the trainer script

    python train_image_classifier.py \
        --train_dir=${TRAIN_DIR} \
        --dataset_name=flowers \
        --dataset_split_name=train \
        --dataset_dir=${DATA_DIR} \
        --model_name=inception_v3 \
        --checkpoint_path=${CHECKPOINT_DIR}/inception_v3.ckpt \
        --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
        --trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
        --max_number_of_steps=1000 \
        --batch_size=32 \
        --learning_rate=0.01 \
        --learning_rate_decay_type=fixed \
        --log_every_n_steps=50 \
        --optimizer=rmsprop \
        --weight_decay=0.00004

This script starts the training process. Training will last for 1000 steps, and every 50 steps progress is logged to console.
This will take like an hour or so, and it's really CPU intensive. (You can also use a GPU if you have one, it's up to you to find out how to do that)

On my Macbook Pro, I ran into following problem

    INFO:tensorflow:Error reported to Coordinator: <class 'tensorflow.python.framework.errors_impl.InvalidArgumentError'>, Cannot assign a device for operation 'gradients/InceptionV3/AuxLogits/Conv2d_1b_1x1/BatchNorm/moments/mean_grad/Prod_1': Operation was explicitly assigned to /device:GPU:0 but available devices are [ /job:localhost/replica:0/task:0/cpu:0 ]. Make sure the device specification refers to a valid device.

This was easily fixed by changing one line in **train_image_classifier.py**

    nano train_image_classifier.py
    ...
    tf.app.flags.DEFINE_boolean('clone_on_cpu', False,
                            'Use CPUs to deploy clones.')
    ->
    tf.app.flags.DEFINE_boolean('clone_on_cpu', True,
                            'Use CPUs to deploy clones.')
    ...

You can find my modified version of **train_image_classifier.py** included in this git repository.
After fixing the problem, try again.

After a long time, the training is completed at 1000th step.

## Evaluating resulting model

Next, let's test the resulting model by evaluating it.

    $ python eval_image_classifier.py \
        --checkpoint_path=${TRAIN_DIR} \
        --eval_dir=${TRAIN_DIR}/eval \
        --dataset_name=flowers \
        --dataset_split_name=validation \
        --dataset_dir=${DATA_DIR} \
        --model_name=inception_v3

The result should look something like this

    INFO:tensorflow:Scale of 0 disables regularizer.
    WARNING:tensorflow:From eval_image_classifier.py:157: streaming_recall_at_k (from tensorflow.contrib.metrics.python.ops.metric_ops) is deprecated and will be removed after 2016-11-08.
    Instructions for updating:
    Please use `streaming_sparse_recall_at_k`, and reshape labels from [batch_size] to [batch_size, 1].
    INFO:tensorflow:Evaluating /tmp/flowers-models/inception_v3/model.ckpt-1000
    INFO:tensorflow:Starting evaluation at 2017-10-02-13:02:25
    2017-10-02 16:02:26.471617: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
    2017-10-02 16:02:26.471649: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
    2017-10-02 16:02:26.471654: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
    2017-10-02 16:02:26.471658: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA     instructions, but these are available on your machine and could speed up CPU computations.
    INFO:tensorflow:Restoring parameters from /tmp/flowers-models/inception_v3/model.ckpt-1000
    INFO:tensorflow:Evaluation [1/4]
    INFO:tensorflow:Evaluation [2/4]
    INFO:tensorflow:Evaluation [3/4]
    INFO:tensorflow:Evaluation [4/4]
    2017-10-02 16:04:02.181376: I tensorflow/core/kernels/logging_ops.cc:79] eval/Accuracy[0.845]
    2017-10-02 16:04:02.181376: I tensorflow/core/kernels/logging_ops.cc:79] eval/Recall_5[1]
    INFO:tensorflow:Finished evaluation at 2017-10-02-13:04:02

This means everything works.

# Serving fine-tuned Inception v3 model

Now we need to make the model deployable. After this part, you'll be able to run the model on a server or even on Google Cloud.

## Creating TensorFlow Saved Model

Assuming you are in serving directory, run the bazel-bin **inception_saved_model**

    $ mkdir ${TRAIN_DIR}/inception-export
    :serving $ bazel-bin/tensorflow_serving/example/inception_saved_model --checkpoint_dir=${TRAIN_DIR} --output_dir=${TRAIN_DIR}/inception-export

Most likely you will get errors. The included **inception_saved_model.py** script is not meant to be run with flowers dataset fine-tuned inception model.
In order to fix this, I have made changed to the **inception_saved_model.py**

Changes made to tensorflow_serving/example/inception_saved_model.py

    diff --git a/tensorflow_serving/example/inception_saved_model.py b/tensorflow_serving/example/inception_saved_model.py
    index 86b20c4..4042b5b 100644
    --- a/tensorflow_serving/example/inception_saved_model.py
    +++ b/tensorflow_serving/example/inception_saved_model.py
    @@ -38,12 +38,12 @@ tf.app.flags.DEFINE_integer('image_size', 299,
                             """Needs to provide same value as in training.""")
     FLAGS = tf.app.flags.FLAGS

    -NUM_CLASSES = 1000
    +NUM_CLASSES = 5
     NUM_TOP_CLASSES = 5

     WORKING_DIR = os.path.dirname(os.path.realpath(__file__))
    -SYNSET_FILE = os.path.join(WORKING_DIR, 'imagenet_lsvrc_2015_synsets.txt')
    -METADATA_FILE = os.path.join(WORKING_DIR, 'imagenet_metadata.txt')
    +SYNSET_FILE = os.path.join(WORKING_DIR, 'flowers_synsets.txt')
    +METADATA_FILE = os.path.join(WORKING_DIR, 'flowers_metadata.txt')


     def export():
    @@ -74,7 +74,7 @@ def export():
         images = tf.map_fn(preprocess_image, jpegs, dtype=tf.float32)

         # Run inference.
    -    logits, _ = inception_model.inference(images, NUM_CLASSES + 1)
    +    logits, _ = inception_model.inference(images, NUM_CLASSES)

         # Transform output to topK result.
         values, indices = tf.nn.top_k(logits, NUM_TOP_CLASSES)
    @@ -83,7 +83,7 @@ def export():
         # the human readable class description for the i'th index.
         # Note that the 0th index is an unused background class
         # (see inception model definition code).
    -    class_descriptions = ['unused background']
    +    class_descriptions = []
         for s in synsets:
           class_descriptions.append(texts[s])
         class_tensor = tf.constant(class_descriptions)
    @@ -92,14 +92,17 @@ def export():
         classes = table.lookup(tf.to_int64(indices))

         # Restore variables from training checkpoint.
    -    variable_averages = tf.train.ExponentialMovingAverage(
    -        inception_model.MOVING_AVERAGE_DECAY)
    -    variables_to_restore = variable_averages.variables_to_restore()
    -    saver = tf.train.Saver(variables_to_restore)
    +    # variable_averages = tf.train.ExponentialMovingAverage(
    +    #     inception_model.MOVING_AVERAGE_DECAY)
    +    # variables_to_restore = variable_averages.variables_to_restore()
    +    # saver = tf.train.Saver(variables_to_restore)
         with tf.Session() as sess:
           # Restore variables from training checkpoints.
           ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
           if ckpt and ckpt.model_checkpoint_path:
    +       init = tf.global_variables_initializer()
    +       sess.run(init)
    +        saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
             saver.restore(sess, ckpt.model_checkpoint_path)
             # Assuming model_checkpoint_path looks something like:
             #   /my-favorite-path/imagenet_train/model.ckpt-0,

These fixed the errors for me.
You can find my modified version of **inception_saved_model.py** included in this git repository.
You will also need the **'flowers_synsets.txt'** and **'flowers_metadata.txt'**. They can be found in the same directory.

Run the above bazel-bin command again.
Result should look something like this:

    :serving $ bazel-bin/tensorflow_serving/example/inception_saved_model --checkpoint_dir=${TRAIN_DIR} --output_dir=${TRAIN_DIR}/inception-export
    2017-10-02 16:37:31.745096: I external/org_tensorflow/tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.2 AVX AVX2 FMA
    Successfully loaded model from /tmp/flowers-models/inception_v3/model.ckpt-1000 at step=1000.
    Exporting trained model to /tmp/flowers-models/inception_v3/inception-export/1
    Successfully exported model to /tmp/flowers-models/inception_v3/inception-export

## Testing the exported model locally

Next we will run tensorflow model server locally, serving the exported fine-tuned Inception model

Run the bazel-bin of **tensorflow_model_server**

    :serving $ bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9000 --model_name=inception --model_base_path=/tmp/flowers-models/inception_v3/inception-export/
    2017-10-02 16:44:00.677688: I tensorflow_serving/model_servers/main.cc:147] Building single TensorFlow model file config:  model_name: inception model_base_path: /tmp/flowers-models/inception_v3/inception-export/
    2017-10-02 16:44:00.678383: I tensorflow_serving/model_servers/server_core.cc:441] Adding/updating models.
    2017-10-02 16:44:00.678396: I tensorflow_serving/model_servers/server_core.cc:492]  (Re-)adding model: inception
    2017-10-02 16:44:00.782665: I tensorflow_serving/core/basic_manager.cc:705] Successfully reserved resources to load servable {name: inception version: 1}
    2017-10-02 16:44:00.782705: I tensorflow_serving/core/loader_harness.cc:66] Approving load for servable version {name: inception version: 1}
    2017-10-02 16:44:00.782718: I tensorflow_serving/core/loader_harness.cc:74] Loading servable version {name: inception version: 1}
    2017-10-02 16:44:00.782774: I external/org_tensorflow/tensorflow/contrib/session_bundle/bundle_shim.cc:360] Attempting to load native SavedModelBundle in bundle-shim from: /tmp/flowers-models/inception_v3/inception-export/1
    2017-10-02 16:44:00.782793: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:236] Loading SavedModel from: /tmp/flowers-models/inception_v3/inception-export/1
    2017-10-02 16:44:00.897964: I external/org_tensorflow/tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.2 AVX AVX2 FMA
    2017-10-02 16:44:01.160579: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:155] Restoring SavedModel bundle.
    2017-10-02 16:44:01.720062: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:190] Running LegacyInitOp on SavedModel bundle.
    2017-10-02 16:44:01.907012: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:284] Loading SavedModel: success. Took 1124208 microseconds.
    2017-10-02 16:44:01.907275: I tensorflow_serving/core/loader_harness.cc:86] Successfully loaded servable version {name: inception version: 1}
    2017-10-02 16:44:01.914783: I tensorflow_serving/model_servers/main.cc:288] Running ModelServer at 0.0.0.0:9000 ...

Now the server is running. Open another terminal tab and let's test the server.

First download a photo which you are going to evaluate.
This downloads a photo of tulips.

    wget https://cloud.google.com/blog/big-data/2016/12/images/148114735559140/image-classification-4.png

Next, run the bazel-bin for inception_client

    :serving $ bazel-bin/tensorflow_serving/example/inception_client --server=0.0.0.0:9000 --image=/<absolute>/<path>/<to>/image-classification-4.png

If everything works, you should be able to get output like this:

    outputs {
      key: "classes"
      value {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
          dim {
            size: 5
          }
        }
        string_val: "tulips"
        string_val: "daisy"
        string_val: "sunflowers"
        string_val: "dandelion"
        string_val: "roses"
      }
    }
    outputs {
      key: "scores"
      value {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 1
          }
          dim {
            size: 5
          }
        }
        float_val: 62680916.0
        float_val: 22133728.0
        float_val: 7779077.5
        float_val: -104189456.0
        float_val: -136751696.0
      }
    }

If you see this output, congratulations! You have successfully retrained and served an Inception v3 model using test dataset.
Now you can start customizing everything to suit your needs.

Can you, for example, make an inception model that can recognize traffic signs?

Want to learn more? Read these:
https://cloud.google.com/ml-engine/docs/how-tos/
https://www.tensorflow.org/get_started/
https://www.tensorflow.org/programmers_guide/
https://www.tensorflow.org/tutorials/
https://www.tensorflow.org/deploy/
