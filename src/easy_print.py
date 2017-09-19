from collections import OrderedDict
import sys
import types
import numpy as np


class easy_print:
    def __init__(self):
        self.phases = {}
        self.default_type_print = {}
        self.col_sep = ' '
        self.file_list = {}

    def add_phase(self, *printed, **params):
        # parse phase
        if params.get('phase'):
            temp_phase = params.get('phase')
        else:
            temp_phase = 'phase'
        if not self.phases.get(temp_phase):
            self.phases[temp_phase] = {}
        if not self.phases[temp_phase].get('data'):
            self.phases[temp_phase]['data'] = OrderedDict()
        # parse file
        if params.get('file'):
            self.phases[temp_phase]['file'] = params.get('file')
            if not self.file_list.get(params.get('file')):
                self.file_list[params.get('file')] = None
        else:
            self.phases[temp_phase]['file'] = sys.stdout
        if params.get('sep'):
            self.phases[temp_phase]['col_sep'] = params.get('sep')
        else:
            self.phases[temp_phase]['col_sep'] = ' '
        if params.get('check_phase'):
            self.phases[temp_phase]['check_phase'] = params.get('check_phase')
        # parse printed simple params
        if printed:
            for i in printed:
                if not self.phases[temp_phase]['data'].get(i):
                    self.phases[temp_phase]['data'][i] = {}
                self.phases[temp_phase]['data'][i]['var'] = self.default_type_print.get(type(i))
                self.phases[temp_phase]['data'][i]['len'] = len(str(i))
        for key in params.keys():
            if key != 'file' and key != 'phase' and key != 'sep' and key != 'check_phase':
                if not self.phases[temp_phase]['data'].get(key):
                    self.phases[temp_phase]['data'][key] = {}
                self.phases[temp_phase]['data'][key]['var'] = params[key]
                self.phases[temp_phase]['data'][key]['len'] = len(str(key))

    def default_print(self, **kwargs):
        for key in kwargs.keys():
            self.default_type_print[key] = kwargs[key]
#    def close(self):
#        for i in self.phases_outfile.values():
#            i.close()

    def print_head(self, data={'var1': 'base_value'}, phase='phase', check_phase=None):
        if self.phases.get(phase):
            temp_file = self.phases[phase]['file']
            temp_result_out = ''
            for key in self.phases[phase]['data'].keys():
                val = data.get(key)
                temp_print_type = self.phases[phase]['data'][key]['var']
                if isinstance(temp_print_type, types.FunctionType):
                    temp_print_type()
                if val == None:
                    val = key
                check_phase = self.phases[phase].get('check_phase')
                if check_phase:
                    temp_len = self.phases[check_phase]['data'][key]['len']
                else:
                    temp_len = self.phases[phase]['data'][key]['len']
                if temp_len < 20:
                    temp_print_type = '{:^%d}' % temp_len
                else:
                    temp_print_type = '{:<%d}' % temp_len
                if type(val) == dict and val.get('length'):
                    if not val.get('form') or val.get('form') == 'all_columns':
                        vect_form = ''
                        for i in range(0, val.get('length')):

                            if val.get('pattern'):
                                vect_pattern = val.get('pattern')
                            else:
                                vect_pattern = 'x{:d}'
                            temp_column = vect_pattern.format(i)
                            temp_column1 = '{: >%d}' % int(temp_len / val.get('length'))
                            vect_form = vect_form + temp_column1.format(temp_column)
                        val = vect_form
                        temp_print_type = ' {:s}'
                temp_result_out = temp_result_out + temp_print_type.format(val) + ' ' + self.phases[phase]['col_sep']
            if temp_file != sys.stdout:
                temp_o_file = open(temp_file, "r+")
                old = temp_o_file.read()
                temp_o_file.seek(0)
                temp_o_file.write(temp_result_out + '\n' + old)
            else:
                print(temp_result_out, self.phases[phase]['col_sep'])
        else:
            print("phase %s doesn't exist" % phase)

    def print_phase(self, phase='phase', data={'var1': 'base_value'}):
        if self.phases.get(phase):
            temp_file = self.phases[phase]['file']
            for key in self.phases[phase]['data'].keys():
                val = data.get(key)
                temp_print_type = self.phases[phase]['data'][key]['var']
                if isinstance(temp_print_type, types.FunctionType):
                    temp_print_type()
                if val is not None:
                    if not temp_print_type:
                        temp_type = str(type(val)).replace('<type ', '')
                        temp_type = temp_type.replace('>', '')
                        temp_print_type = self.default_type_print.get(temp_type)
                    if type(temp_print_type) == str:
                        temp_result_out = temp_print_type.format(val)
                    else:
                        temp_result_out = val
                    self.phases[phase]['data'][key]['len'] = len(str(temp_result_out))
                else:
                    check_phase = self.phases[phase].get('check_phase')
                    if check_phase:
                        temp_print_type = '{:>%d}' % self.phases[check_phase]['data'][key]['len']
                    else:
                        temp_print_type = '{:>%d}' % self.phases[phase]['data'][key]['len']
                    temp_result_out = temp_print_type.format('')
                if temp_file != sys.stdout:
                    if self.file_list[temp_file]:
                        temp_o_file = open(temp_file, 'a')
                    else:
                        temp_o_file = open(temp_file, 'w')
                        self.file_list[temp_file] = 1
                print(temp_result_out, self.phases[phase]['col_sep'], file=temp_o_file, end='')
            print('\n', file=temp_o_file, end='')

        else:
            print("phase %s doesn't exist" % phase)

# easy = easy_print()
# easy.add_phase('a', 'b', 'c', 'd', phase = 'test', file = 'test.txt', c = lambda: print('print_test'), b = '{:>6}', a = '{: 5.3e}', sep = ' | ')
# easy.default_print(int = '{:11.11e}')
# easy.print_phase('test', {'a' : 1, 'b' : 2, 'c' : 3})
