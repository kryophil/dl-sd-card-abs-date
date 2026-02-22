# MIT License
#
# Copyright (c) 2016 Decentlab GmbH
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import json
import warnings
import copy
import re
import requests
import pandas as pd

def query(domain, api_key, time_filter='',
          device='//', location='//',
          sensor='//', include_network_sensors=False,
          channel='//',
          agg_func=None, agg_interval=None,
          do_unstack=True,
          convert_timestamp=True,
          timezone='UTC',
          with_location=False,
          database='main'):

    select_var = 'value'
    fill = ''
    interval = ''

    if agg_func is not None:
        select_var = agg_func + '("value") as value'
        fill = 'fill(null)'

    if agg_interval is not None:
        interval = ', time({})'.format(agg_interval)

    if time_filter != '':
        time_filter = ' AND ' + time_filter

    filter_ = (' location =~ {}'
               ' AND node =~ {}'
               ' AND sensor =~ {}'
               ' AND ((channel =~ {} OR channel !~ /.+/)'
               ' {})').format(location,
                              device,
                              sensor,
                              channel,
                              ('' if include_network_sensors
                               else 'AND channel !~ /^link-/'))

    q = ('SELECT {} FROM "measurements" '
         ' WHERE {} {}'
         ' GROUP BY channel,node,sensor,unit{},uqk,title'
         ' {} {}').format(select_var,
                          filter_,
                          time_filter,
                          ',location' if with_location else '',
                          interval,
                          fill)

    URL = 'https://{}/api/datasources/proxy/uid/{}/query'.format(domain,
                                                                  database)

    r = requests.get(URL,
                     params={'db': 'main',
                             'epoch': 'ms',
                             'q': q},
                     headers={'Authorization': 'Bearer {}'.format(api_key)})

    data = json.loads(r.text)

    if 'results' not in data or 'series' not in data['results'][0]:
        raise ValueError("No series returned: {}".format(r.text))

    def _ix2df(series):
        df = pd.DataFrame(series['values'], columns=series['columns'])
        df['series'] = series['tags']['uqk']
        if with_location:
            df['location'] = series['tags']['location']
        return df, (series['tags']['uqk'], series['tags'])

    series, tags = zip(*(_ix2df(s)
                         for r in data['results']
                         for s in r['series']))

    df = pd.concat(series)
    tags = dict(tags)

    if convert_timestamp:
        df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True)
        try:
            df['time'] = df['time'].dt.tz_localize('UTC')
        except TypeError:
            pass
        df['time'] = df['time'].dt.tz_convert(timezone)

    indices = ['time', 'series']
    if with_location:
        indices.append('location')

    df = df.set_index(indices)
    df = df.sort_index()

    if do_unstack:
        df = df.unstack(level='series')
        if with_location:
            df = df.unstack(level='location')
        df.columns = df.columns.droplevel(0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        setattr(df, 'tags', tags)

    return df
