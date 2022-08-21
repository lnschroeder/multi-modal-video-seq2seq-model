"""kth dataset."""
import os
import re
import imageio
import tensorflow_datasets as tfds

# TODO(kth): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(kth): BibTeX citation
_CITATION = """
"""

# Dataset divided according to the instruction at:
# http://www.nada.kth.se/cvap/actions/00sequences.txt
TRAIN_PEOPLE_IDS = [11, 12, 13, 14, 15, 16, 17, 18]
VAL_PEOPLE_IDS = [19, 20, 21, 23, 24, 25, 1, 4]
TEST_PEOPLE_IDS = [22, 2, 3, 5, 6, 7, 8, 9, 10]

# Dataset divided as in Stochastic Video Generation with a Learned Prior
# Stochastic Video Generation with a Learned Prior
SVG_TRAIN_PEOPLE_IDS = list(range(1, 21))
SVG_TEST_PEOPLE_IDS = list(range(21, 26))


def parse_sequence_file(path):
  frame_indices = {}

  with open(path, 'r') as fileobject:
    for row in fileobject:
      if 'frames' in row:
        row = re.sub("[\t\n ]", "", row)
        current_filename, subsequences = row.split('frames')
        subsequences = subsequences.split(',')
        current_filename +=  "_uncomp.avi"
        frame_indices[current_filename] = [(int(s.split('-')[0]), int(s.split('-')[1])) for s in subsequences]

  return frame_indices


def split_videos(path_sequence_file, path_video_files, path_output_files):

  frame_indices = parse_sequence_file(path_sequence_file)
  file_names = sorted(os.listdir(path_video_files))

  if not os.path.exists(path_output_files):
    os.makedirs(path_output_files)

  for fname in file_names:
    path_file_in = os.path.join(path_video_files, fname)
    reader = imageio.get_reader(path_file_in, 'ffmpeg', fps=25)

    for i, seq in enumerate(frame_indices[fname]):
      path_seq = os.path.join(path_output_files, fname[:-11]+"_"+str(i)+'.avi')
      with imageio.get_writer(path_seq, fps=25, macro_block_size=1) as writer:
        # for some reason sometimes the sequence is shorter than the subsequence limits
        for j, frame in enumerate(reader.iter_data()):
          if seq[0] <= j and j <= seq[1]:
            writer.append_data(frame)


class KthConfig(tfds.core.BuilderConfig):
  """"Configuration for KTH video rescaling."""

  def __init__(self, width=None, height=None, seq_splits=True, **kwargs):
    """The parameters specifying how the dataset will be processed.

    Args:
      width: An integer with the width or None.
      height: An integer with the height or None.
      **kwargs: Passed on to the constructor of `BuilderConfig`.
    """
    VERSION = tfds.core.Version('1.0.2')
    RELEASE_NOTES = {
      '1.0.2': 'Initial release + SVG split + subsplit option.',
    }

    super(KthConfig, self).__init__(
        version=VERSION,
        release_notes=RELEASE_NOTES,
        **kwargs,
    )

    self.width = width or height
    self.height = height or width
    self.seq_splits = seq_splits


class Kth(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for kth dataset."""

  BUILDER_CONFIGS = [
      KthConfig(
          name='default',
          description='KTH dataset default frame size.',
          width=None,
          height=None
      ),
      KthConfig(
          name='default_no_seqsplits',
          description='KTH dataset default frame size, no splits of sequences.',
          width=None,
          height=None,
          seq_splits=False
      ),
      KthConfig(
          name='small',
          description='KTH dataset small frame size.',
          width=64,
          height=64
      ),
      KthConfig(
          name='small_no_seqsplits',
          description='KTH dataset small frame size, no splits of sequences.',
          width=64,
          height=64,
          seq_splits=False
      ),
      KthConfig(
          name='large',
          description='KTH dataset large frame size.',
          width=128,
          height=128
      ),
      KthConfig(
          name='large_no_seqsplits',
          description='KTH dataset large frame size, no splits of sequences.',
          width=128,
          height=128,
          seq_splits=False
      )
  ]


  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""

    if self.builder_config.width is not None and self.builder_config.height is not None:
      ffmpeg_extra_args = ('-vf', 'scale={}x{}'.format(self.builder_config.height, self.builder_config.width))
      video_shape = (None, self.builder_config.height, self.builder_config.width, 3)
    else:
      ffmpeg_extra_args = []
      video_shape = (None, 160, 120, 3)

    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({

          'video': tfds.features.Video(
            video_shape,
            ffmpeg_extra_args=ffmpeg_extra_args,
            encoding_format='jpeg'
          ),

          'label': tfds.features.ClassLabel(
              names=[
                "boxing",
                "handclapping",
                "handwaving",
                "jogging",
                "running",
                "walking"
              ]
            ),

        }),

        supervised_keys=('video', 'label'),
        homepage='https://www.csc.kth.se/cvap/actions/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    paths = dl_manager.download_and_extract(
      [
        'https://tubcloud.tu-berlin.de/s/R7ZLTynPonfGdq9/download',
        'https://tubcloud.tu-berlin.de/s/qjPm4kTtnrJq5QA/download',
        'https://tubcloud.tu-berlin.de/s/pksmssKiqHaQLwa/download',
        'https://tubcloud.tu-berlin.de/s/fLytZAdyRRMN72t/download',
        'https://tubcloud.tu-berlin.de/s/t9PWjjgXxbc4AXt/download',
        'https://tubcloud.tu-berlin.de/s/8NRzXA9exA8Pqrt/download',
        'https://tubcloud.tu-berlin.de/s/PzfzSTnCQLpcCsF/download'
      ]
    )
    #   [
    #     'https://www.csc.kth.se/cvap/actions/00sequences.txt',
    #     'http://www.nada.kth.se/cvap/actions/walking.zip',
    #     'http://www.nada.kth.se/cvap/actions/jogging.zip',
    #     'http://www.nada.kth.se/cvap/actions/running.zip',
    #     'http://www.nada.kth.se/cvap/actions/boxing.zip',
    #     'http://www.nada.kth.se/cvap/actions/handwaving.zip',
    #     'http://www.nada.kth.se/cvap/actions/handclapping.zip'
    #   ]
    # )

    # Either splitting videos as in the original publication or not.
    if self.builder_config.seq_splits:
      path_subsequences = os.path.join(os.path.dirname(paths[1]), 'kth_subsequences')
      for path in paths[1:]:
        split_videos(paths[0], path, path_subsequences)
      paths_sequences = sorted(os.listdir(path_subsequences))
      paths_sequences = [os.path.join(path_subsequences, f) for f in paths_sequences]

    else:
      paths_sequences = []
      for path in paths[1:]:
        paths = sorted(os.listdir(path))
        paths = [os.path.join(path, f) for f in paths]
        paths_sequences += paths

    return {
        'train': self._generate_examples(paths_sequences, 'train'),
        'val': self._generate_examples(paths_sequences, 'val'),
        'trainval': self._generate_examples(paths_sequences, 'trainval'),
        'test': self._generate_examples(paths_sequences, 'test'),
        'svg_train': self._generate_examples(paths_sequences, 'svg_train'),
        'svg_trainval': self._generate_examples(paths_sequences, 'svg_train'),
        'svg_test': self._generate_examples(paths_sequences, 'svg_test')
    }

  def _generate_examples(self, paths_video_files, split):
    """Yields examples."""
    if split == "svg_train":
      IDS = SVG_TRAIN_PEOPLE_IDS
    elif split == "svg_test":
      IDS = SVG_TEST_PEOPLE_IDS
    elif split == "train":
      IDS = TRAIN_PEOPLE_IDS
    elif split == "val":
      IDS = VAL_PEOPLE_IDS
    elif split == "trainval":
      IDS = TRAIN_PEOPLE_IDS+VAL_PEOPLE_IDS
    elif split == "test":
      IDS = TEST_PEOPLE_IDS

    file_names = [f for f in paths_video_files if int(os.path.basename(f).split("_")[0][6:]) in IDS]

    for f in file_names:
      yield os.path.basename(f), {
          'video': f,
          'label': os.path.basename(f).split("_")[1],
      }
