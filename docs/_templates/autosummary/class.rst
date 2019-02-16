{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   {% if methods %}
   .. rubric:: Methods

   .. autosummary::
      :toctree:
   {% for item in methods %}
      {% if item not in ['__init__', 'clear', 'fromkeys', 'items', 'keys', 'pop', 'popitem', 'setdefault', 'update', 'values', 'getfield', 'partition', 'all', 'any', 'argmax', 'argmin', 'argpartition', 'argsort', 'astype', 'byteswap', 'choose', 'clip', 'compress', 'conj', 'conjugate', 'cumprod', 'cumsum', 'diagonal', 'dot', 'dump', 'dumps', 'fill', 'flatten', 'item', 'itemset', 'max', 'mean', 'min', 'newbyteorder', 'nonzero', 'portition', 'prod', 'ptp', 'put', 'ravel', 'repeat', 'reshape', 'resize', 'round', 'searchsorted', 'setfield', 'setflags', 'sort', 'squeeze', 'std', 'sum', 'swapaxes', 'take', 'tobytes', 'tofile', 'tolist', 'tostring', 'trace', 'transpose', 'var', 'view'] %}~{{ name }}.{{ item }}{% endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}
