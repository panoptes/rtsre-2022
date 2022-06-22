import re
import warnings
import traceback
from enum import IntEnum, auto
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Pattern, Union, Optional, Tuple

from dateutil.parser import parse as parse_date
from dateutil.tz import UTC
import pandas as pd
from tqdm.auto import tqdm
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.io.fits.header import Header
from astropy.utils.data import download_file
from astropy.nddata import CCDData, Cutout2D
from astropy.wcs import WCS, FITSFixedWarning

from loguru import logger

from panoptes.utils.utils import listify
from panoptes.utils.time import current_time, flatten_time
from panoptes.utils.images import fits as fits_utils


warnings.filterwarnings('ignore', category=FITSFixedWarning)


class SequenceStatus(IntEnum):
    RECEIVING = 0
    RECEIVED = 10


class ImageStatus(IntEnum):
    ERROR = -10
    MASKED = -5
    UNKNOWN = -1
    RECEIVING = 0
    RECEIVED = 5
    CALIBRATING = 10
    CALIBRATED = 15
    SOLVING = 20
    SOLVED = 25
    MATCHING = 30
    MATCHED = 35
    EXTRACTING = 40
    EXTRACTED = 45


class ObservationStatus(IntEnum):
    ERROR = -10
    NOT_ENOUGH_FRAMES = -2
    UNKNOWN = -1
    CREATED = 0
    CALIBRATED = 10
    MATCHED = 20
    PROCESSING = 30
    PROCESSED = 35


IMG_BASE_URL = 'https://storage.googleapis.com/'
OBS_BASE_URL = 'https://storage.googleapis.com/panoptes-observations'
IMG_METADATA_URL = 'https://us-central1-panoptes-exp.cloudfunctions.net/get-observation-metadata'
OBSERVATIONS_URL = 'https://storage.googleapis.com/panoptes-exp.appspot.com/observations.csv'

PATH_MATCHER: Pattern[str] = re.compile(r"""^
                                (?P<pre_info>.*)?                       # Anything before unit_id
                                (?P<unit_id>PAN\d{3})                   # unit_id   - PAN + 3 digits
                                /?(?P<field_name>.*)?                   # Legacy field name - any                                
                                /(?P<camera_id>[a-gA-G0-9]{6})          # camera_id - 6 digits
                                /(?P<sequence_time>[0-9]{8}T[0-9]{6})   # Observation start time
                                /(?P<image_time>[0-9]{8}T[0-9]{6})      # Image start time
                                (?P<post_info>.*)?                      # Anything after (file ext)
                                $""",
                                        re.VERBOSE)


@dataclass
class ObservationPathInfo:
    """Parse the location path for an image.

    This is a small dataclass that offers some convenience methods for dealing
    with a path based on the image id.

    This would usually be instantiated via `path`:

    ..doctest::

        >>> from panoptes.pipeline.utils.metadata import ObservationPathInfo
        >>> bucket_path = 'gs://panoptes-images-background/PAN012/Hd189733/358d0f/20180824T035917/20180824T040118.fits'
        >>> path_info = ObservationPathInfo(path=bucket_path)

        >>> path_info.id
        'PAN012_358d0f_20180824T035917_20180824T040118'

        >>> path_info.unit_id
        'PAN012'

        >>> path_info.sequence_id
        'PAN012_358d0f_20180824T035917'

        >>> path_info.image_id
        'PAN012_358d0f_20180824T040118'

        >>> path_info.as_path(base='/tmp', ext='.jpg')
        '/tmp/PAN012/358d0f/20180824T035917/20180824T040118.jpg'

        >>> ObservationPathInfo(path='foobar')
        Traceback (most recent call last):
          ...
        ValueError: Invalid path received: self.path='foobar'


    """
    unit_id: str = None
    camera_id: str = None
    field_name: str = None
    sequence_time: Union[str, datetime, Time] = None
    image_time: Union[str, datetime, Time] = None
    path: Union[str, Path] = None

    def __post_init__(self):
        """Parse the path when provided upon initialization."""
        if self.path is not None:
            path_match = PATH_MATCHER.match(self.path)
            if path_match is None:
                raise ValueError(f'Invalid path received: {self.path}')

            self.unit_id = path_match.group('unit_id')
            self.camera_id = path_match.group('camera_id')
            self.field_name = path_match.group('field_name')
            self.sequence_time = Time(parse_date(path_match.group('sequence_time')))
            self.image_time = Time(parse_date(path_match.group('image_time')))

    @property
    def id(self):
        """Full path info joined with underscores"""
        return self.get_full_id()

    @property
    def sequence_id(self) -> str:
        """The sequence id."""
        return f'{self.unit_id}_{self.camera_id}_{flatten_time(self.sequence_time)}'

    @property
    def image_id(self) -> str:
        """The matched image id."""
        return f'{self.unit_id}_{self.camera_id}_{flatten_time(self.image_time)}'

    def as_path(self, base: Union[Path, str] = None, ext: str = None) -> Path:
        """Return a Path object."""
        image_str = flatten_time(self.image_time)
        if ext is not None:
            image_str = f'{image_str}.{ext}'

        full_path = Path(self.unit_id, self.camera_id, flatten_time(self.sequence_time), image_str)

        if base is not None:
            full_path = base / full_path

        return full_path

    def get_full_id(self, sep='_') -> str:
        """Returns the full path id with the given separator."""
        return f'{sep}'.join([
            self.unit_id,
            self.camera_id,
            flatten_time(self.sequence_time),
            flatten_time(self.image_time)
        ])

    @classmethod
    def from_fits(cls, fits_file):
        header = fits_utils.getheader(fits_file)
        return cls.from_fits_header(header)

    @classmethod
    def from_fits_header(cls, header):
        try:
            new_instance = cls(path=header['FILENAME'])
        except ValueError:
            sequence_id = header['SEQID']
            image_id = header['IMAGEID']
            unit_id, camera_id, sequence_time = sequence_id.split('_')
            _, _, image_time = image_id.split('_')

            new_instance = cls(unit_id=unit_id,
                               camera_id=camera_id,
                               sequence_time=Time(parse_date(sequence_time)),
                               image_time=Time(parse_date(image_time)))

        return new_instance


class ObservationInfo():
    def __init__(self, sequence_id=None, meta=None):
        """Initialize the observation info with a sequence_id"""
        if meta is not None:
            self.sequence_id = meta.sequence_id
            self.meta = meta
        else:
            self.sequence_id = sequence_id
            self.meta = dict()

        self.image_metadata = self.get_metadata()
        self.raw_images = self.get_image_list()
        self.processed_images = self.get_image_list(raw=False)

        
    def get_image_data(self, idx=0, coords=None, box_size=None, use_raw=True):
        """Downloads the image data for the given index."""
        
        if use_raw:
            image_list = self.raw_images
        else:
            image_list = self.processed_images
            
        data_img = image_list[idx]
        wcs_img = self.processed_images[idx]
        
        data0, header0 = fits_utils.getdata(data_img, header=True)
        wcs0 = fits_utils.getwcs(wcs_img)
        ccd0 = CCDData(data0, wcs=wcs0, unit='adu', meta=header0)
            
        if coords is not None and box_size is not None:
            ccd0 = Cutout2D(ccd0, coords, box_size)

        return ccd0

    def get_metadata(self):
        """Download the image metadata associated with the observation."""
        images_df = pd.read_csv(f'{IMG_METADATA_URL}?sequence_id={self.sequence_id}')

        # Set a time index.
        images_df.time = pd.to_datetime(images_df.time)
        images_df = images_df.set_index(['time']).sort_index()

        print(f'Found {len(images_df)} images in observation')

        return images_df


    def get_image_list(self, raw=True):
        """Get the images for the observation."""
        if raw:
            bucket = 'panoptes-images-raw'
            file_ext = '.fits.fz'
        else:
            bucket = 'panoptes-images-processed'
            file_ext = '-reduced.fits.fz'

        image_list = [IMG_BASE_URL + bucket + '/' + str(s).replace("_", "/") + file_ext for s in self.image_metadata.uid.values]

        return image_list

    def __str__(self):
        return f'Obs: seq_id={self.sequence_id} num_images={len(self.raw_images)}'

    def __repr__(self):
        return self.meta
                

def search_observations(
        coords=None,
        unit_id=None,
        start_date=None,
        end_date=None,
        ra=None,
        dec=None,
        radius=10,  # degrees
        status='CREATED',
        min_num_images=1,
        source_url=OBSERVATIONS_URL,
        source=None,
        ra_col='coordinates_mount_ra',
        dec_col='coordinates_mount_dec',
):
    """Search PANOPTES observations.

    Either a `coords` or `ra` and `dec` must be specified for search to work.

    >>> from astropy.coordinates import SkyCoord
    >>> from panoptes.pipeline.utils.metadata import search_observations
    >>> coords = SkyCoord.from_name('Andromeda Galaxy')
    >>> start_date = '2019-01-01'
    >>> end_date = '2019-12-31'
    >>> search_results = search_observations(coords=coords, min_num_images=10, start_date=start_date, end_date=end_date)
    >>> # The result is a DataFrame you can further work with.
    >>> image_count = search_results.groupby(['unit_id', 'field_name']).num_images.sum()
    >>> image_count
    unit_id  field_name
    PAN001   Andromeda Galaxy     378
             HAT-P-19             148
             TESS_SEC17_CAM02    9949
    PAN012   Andromeda Galaxy      70
             HAT-P-16 b           268
             TESS_SEC17_CAM02    1983
    PAN018   TESS_SEC17_CAM02     244
    Name: num_images, dtype: Int64
    >>> print('Total minutes exposure:', search_results.total_minutes_exptime.sum())
    Total minutes exposure: 20376.83

    Args:
        coords (`astropy.coordinates.SkyCoord`|None): A valid coordinate instance.
        ra (float|None): The RA position in degrees of the center of search.
        dec (float|None): The Dec position in degrees of the center of the search.
        radius (float): The search radius in degrees. Searches are currently done in
            a square box, so this is half the length of the side of the box.
        start_date (str|`datetime.datetime`|None): A valid datetime instance or `None` (default).
            If `None` then the beginning of 2018 is used as a start date.
        end_date (str|`datetime.datetime`|None): A valid datetime instance or `None` (default).
            If `None` then today is used.
        unit_id (str|list|None): A str or list of strs of unit_ids to include.
            Default `None` will include all.
        status (str|list|None): A str or list of observation status to include.
            Defaults to "matched" for observations that have been fully processed. Passing
            `None` will return all status.
        min_num_images (int): Minimum number of images the observation should have, default 1.
        source_url (str): The remote url where the static CSV file is located, default to PANOPTES
            storage location.
        source (`pandas.DataFrame`|None): The dataframe to use or the search.
            If `None` (default) then the `source_url` will be used to look up the file.

    Returns:
        `pandas.DataFrame`: A table with the matching observation results.
    """

    logger.debug(f'Setting up search params')

    if coords is None:
        try:
            coords = SkyCoord(ra=ra, dec=dec, unit='degree')
        except ValueError:
            raise

            # Setup defaults for search.
    if start_date is None:
        start_date = '2018-01-01'

    if end_date is None:
        end_date = current_time()

    with suppress(TypeError):
        start_date = parse_date(start_date).replace(tzinfo=None)
    with suppress(TypeError):
        end_date = parse_date(end_date).replace(tzinfo=None)

    ra_max = (coords.ra + (radius * u.degree)).value
    ra_min = (coords.ra - (radius * u.degree)).value
    dec_max = (coords.dec + (radius * u.degree)).value
    dec_min = (coords.dec - (radius * u.degree)).value

    logger.debug(f'Getting list of observations')

    # Get the observation list
    obs_df = source
    if obs_df is None:
        local_path = download_file(source_url,
                                   cache='update',
                                   show_progress=False,
                                   pkgname='panoptes')
        obs_df = pd.read_csv(local_path)

    logger.info(f'Found {len(obs_df)} total observations')

    # Perform filtering on other fields here.
    logger.debug(f'Filtering observations')
    obs_df.query(
        f'{dec_col} >= {dec_min} and {dec_col} <= {dec_max}'
        ' and '
        f'{ra_col} >= {ra_min} and {ra_col} <= {ra_max}'
        ' and '
        f'time >= "{start_date}"'
        ' and '
        f'time <= "{end_date}"'
        ' and '
        f'num_images >= {min_num_images}'
        ,
        inplace=True
    )
    logger.debug(f'Found {len(obs_df)} observations after initial filter')

    unit_ids = listify(unit_id)
    if len(unit_ids) > 0 and unit_ids != 'The Whole World! 🌎':
        obs_df.query(f'unit_id in {listify(unit_ids)}', inplace=True)
    logger.debug(f'Found {len(obs_df)} observations after unit filter')

    if status is not None:
        obs_df.query(f'status in {listify(status)}', inplace=True)
    logger.debug(f'Found {len(obs_df)} observations after status filter')

    logger.debug(f'Found {len(obs_df)} observations after filtering')

    obs_df = obs_df.reindex(sorted(obs_df.columns), axis=1)
    obs_df.sort_values(by=['time'], inplace=True)

    # Make sure we show an average exptime.
    obs_df.exptime = obs_df.total_exptime / obs_df.num_images

    # Fix bad names and drop useless columns.
    obs_df = obs_df.rename(columns=dict(camera_camera_id='camera_id'))
    obs_df = obs_df.drop(columns=['received_time', 'urls', 'status'])

    obs_df.time = pd.to_datetime(obs_df.time)

    # Fix bad field name.
    obs_df.loc[obs_df.query('field_name >= "00:00:42+00:00"').index, 'field_name'] = 'M42'

    logger.success(f'Returning {len(obs_df)} observations')
    return obs_df
