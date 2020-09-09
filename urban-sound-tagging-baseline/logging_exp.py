#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""                  
                            borrowed from dnai
    logging.py:

      A class for interacting with Neptune and Comet experiment loggers.

-._    _.--'"`'--._    _.--'"`'--._    _.--'"`'--._    _   
    '-:`.'|`|"':-.  '-:`.'|`|"':-.  '-:`.'|`|"':-.  '.` : '.   
  '.  '.  | |  | |'.  '.  | |  | |'.  '.  | |  | |'.  '.:   '.  '.
  : '.  '.| |  | |  '.  '.| |  | |  '.  '.| |  | |  '.  '.  : '.  `.
  '   '.  `.:_ | :_.' '.  `.:_ | :_.' '.  `.:_ | :_.' '.  `.'   `.
         `-..,..-'       `-..,..-'       `-..,..-'       `         `
"""
#import neptune
from comet_ml import Experiment
#import neptune

class experiment_logger:
    '''
    Interface for logging experiments on neptune, comet, or both.
    Args: log_backend, project_name)
    Other backends may also be added in the future
    Currently defined methods:
        add_params:
        add_tags:
        log_text: strings
        log_metrics: numerical values
        log_figure: pyplot figures
        
        stop: end logging and close connection
    '''
    def __init__(self, log_backend='comet', project_name='idm-ust'):
        '''

        Parameters
        ----------
        log_backend : STR
            One of 'comet', 'neptune', 'all'
        project_name : STR
            
        Returns
        -------
        None.

        '''
        self.proj_name = project_name
        self.backend = log_backend
        #Bool indicating wether neptune logging is enabled
        self.neptune = log_backend=='neptune' or log_backend=='all'
        #Bool indicating wether comet logging is enabled
        self.comet = log_backend=='comet' or log_backend=='all'
        if self.neptune:
            neptune.init("ihounie/"+project_name, 
                         api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiMWYzMzhjMjItYjczNC00NzZhLWFlZTYtOTI2NzE5MzUwZmNkIn0=')
            print("logging experiments on neptune project "+project_name)
            neptune.create_experiment()
        if self.comet:
            self.comet_experiment = Experiment(api_key="ev1VO91pYFVRukLChujjpCdmI",
                            project_name=project_name, workspace="ihounie")
            print("logging experiments on comet project "+project_name)
    
    def add_params(self, params, step=None ):
        '''
        Adds parameters to experiment log

        Parameters
        ----------
        params : Dict
            Key-Value pairs

        Returns
        -------
        None.

        '''
        if self.neptune:
            for key, value in params.items():
                neptune.set_property(key, value)   
            if step is not None:
               neptune.set_property('step', step)
        if self.comet:
            self.comet_experiment.log_parameters(params,step=step)
    def add_tags(self, tags):
        '''
        Adds parameters to experiment log

        Parameters
        ----------
        params : tags
            list of tags (strings)
            e.g.: ['tag1', 'tag2']
            
        Returns
        -------
        None.

        '''
        if self.neptune:
            neptune.append_tag(tags)   
        if self.comet:
            self.comet_experiment.add_tags(tags)
 
    def log_metrics(self, name, value, epoch=None):
        '''
        Logging pointwise metrics

        Parameters
        ----------
        name : STR
            Metric key
        value : Float/Integer/(Boolean/String)
            Comet also allows Boolean/string
            Tuples are lallowed
        epoch: (OPT)  INT
            Epoch - or anything used as x axis when plotting metrics

        Returns
        -------
        None.

        '''
        if self.neptune:
            try:
                if epoch is not None:
                    if type(value) is tuple:
                        print("Logging tuple as r and p-value")
                        for val, n in zip(value, [" (r)", " (p-val)"]):
                            neptune.log_metric(name  + n,epoch,y=val)
                    else:
                        neptune.log_metric(name, epoch, y=value)
                else:
                    if type(value) is tuple:
                        print("Logging tuple as r and p-value")
                        for val, n in zip(value, [" (r)", " (p-val)"]):
                            neptune.log_metric(name+n, val)
                    else:
                        neptune.log_metric(name, value)
            except:
                print("Metric type {} not supported by neptune.".format(type(value)))
                print("logging as text")
                self.log_text( "{}".format(value), key=name)
                
        if self.comet:    
            try:
                if epoch is not None:
                    if type(value) is tuple:
                        print("Logging tuple as r and p-value")
                        for val, n in zip(value, [" (r)", " (p-val)"]):
                            self.comet_experiment.log_metric(name+n, val, step=int(epoch))
                    else:
                        self.comet_experiment.log_metric(name, value, epoch=epoch)
                else:
                    if type(value) is tuple:
                        print("Logging tuple as r and p-value")
                        for val, n in zip(value, [" (r)", " (p-val)"]):
                            self.comet_experiment.log_metric(name+n, val)
                    else:
                        self.comet_experiment.log_metric(name, value)
            except:
                print("Metric type {} not supported by comet.".format(type(value)))
                if type(value) is tuple:
                    print("Logging tuple as x-y pairs")
                    for idx, val in enumerate(value):
                        self.comet_experiment.log_metric(name, val, epoch=idx) 
                else:
                    print("Logging as other.")
                    self.comet_experiment.log_other(name, value)
                
    def log_text(self, string, key=None, epoch=None):
        '''
          Logs text strings

          Parameters
          ----------
          string : STR
              text to  log
          key: STR
              log_name needed for Neptune strings 
          epoch: INT
              epoch or any other index
          
          Returns
          -------
          None.

        '''
        if self.neptune:
            if type(string) is str:
                if key is None:
                    print('Neptune log_name needed for logging text')
                    print('Using a dummy name: text')
                    neptune.log_text('text', string)
                if epoch is None:
                    neptune.log_text(key, string)
                else:
                    neptune.log_text(key, epoch, y=string)        
            else:
                print("Wrong type: logging text must be a string")
        if self.comet:                
            if type(string) is str:
                if key is not None:
                    print("Commet text logging does not  support keys, prepending it to text")
                    string = key+ ', '+string
                if epoch is None:
                    self.comet_experiment.log_text(string)
                else:
                    self.comet_experiment.log_text(string, step=epoch)
            else:
                print("Wrong type: logging text must be a string")
        
    def log_figure(self, figure=None, figure_name=None, step=None):
        '''
        Logs pyplot figure

        Parameters
        ----------
        figure : pyplot figure, optional in comet mandatory in neptune.
            The default is None, uses global pyplot figure.
        figure_name : STR, optional in comet mandatory in neptune.
             The default is None.
        step : INT, optional
            An index. The default is None.

        Returns
        -------
        None.

        '''
        if self.neptune:
            if figure is not None:
                if figure_name is None:
                    print("Figure name must be given to neptune logger")
                    print("Using dummy name: figure")
                    figure_name = 'figure'
                if step is None:
                    neptune.log_image(figure_name, figure)
                else:
                    neptune.log_image(figure_name, step, y=figure)    
            else:
                print("A figure must be passed to neptune logger")
        if self.comet:    
            self.comet_experiment.log_figure(figure_name=figure_name, figure=figure, step=step) 
    def stop(self):
        if self.neptune:
            neptune.stop()
        if self.comet:
            self.comet_experiment.end()
        
    def add_table(self, filename, tabular_data=None, headers=False):
        
        self.comet_experiment.log_table(filename, tabular_data, headers)
        
    def log_image(self, image=None, figure_name=None, step=None):
        '''
        Logs pyplot figure

        Parameters
        ----------
        figure : pyplot figure, optional in comet mandatory in neptune.
            The default is None, uses global pyplot figure.
        figure_name : STR, optional in comet mandatory in neptune.
             The default is None.
        step : INT, optional
            An index. The default is None.

        Returns
        -------
        None.

        '''
        self.log_image(image, name=figure_name, overwrite=False, image_format="png", image_scale=1.0, \
                       image_shape=None, image_colormap=None, image_minmax=None, image_channels="last", \
                       copy_to_tmp=True, step=step)
    
    
    
    
    def log_hist3d(self, values=None, figure_name=None, step=None):
        '''
        Logs pyplot figure
    
        Parameters
        ----------
        figure : pyplot figure, optional in comet mandatory in neptune.
            The default is None, uses global pyplot figure.
        figure_name : STR, optional in comet mandatory in neptune.
             The default is None.
        step : INT, optional
            An index. The default is None.
    
        Returns
        -------
        None.
    
        '''
        if self.neptune:
            print("not implemented")    
        if self.comet:    
            self.comet_experiment.log_histogram_3d(values, name=figure_name, step=step) 
    
    
    def log_table(self, name=None, data=None, headers=False):
        '''
        

        Parameters
        ----------
        name : str
            Table name
        data : array, list
            
        headers : TYPE, optional
            wether to use headers

        Returns
        -------
        None.

        '''
        if self.neptune:
            print("not implemented")
        if self.comet:
            self.comet_experiment.log_table(name+'.csv', tabular_data= data, headers = headers )
        
    
    
    def end(self):
        '''
        Ends experiment logging without interrupting python programme
        Parameters
        ----------
        Returns
        ------
        '''
        if self.neptune:
            print("not implemented")
        if self.comet:
            self.comet_experiment.end()
