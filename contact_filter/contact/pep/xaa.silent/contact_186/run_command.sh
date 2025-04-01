#!/bin/bash
../../../../../software/contact/rosetta_scripts -parser:protocol ../../../../../software/contact/just_contact_patch.xml -parser:script_vars patchdock_res=186 -beta_nov16 -overwrite -out:file:scorefile score.sc -out:file:silent_struct_type binary -out:file:silent /dev/null -in:file:silent ../../../xaa.silent
