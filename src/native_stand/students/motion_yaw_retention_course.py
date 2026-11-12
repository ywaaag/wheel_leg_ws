"""Fixed yaw stage with explicit old-command source-policy retention; never promotes."""
from . import motion_yaw_width_course as base
VERSION='motion31_yaw_source_retention_fixed_v1'
ASSESSMENT=base.ASSESSMENT
TARGETS=base.TARGETS
commands=base.commands
stage_config=base.stage_config
cases=base.cases
effective_spec=base.effective_spec
advance=base.advance
choose=base.choose
extend_mixture=base.extend_mixture
protocol=base.protocol

def configuration():
 c=base.configuration();c.update(version=VERSION,scope='fixed±.5yaw; old-command cohort-balanced source policy retention; no promotion')
 return c
