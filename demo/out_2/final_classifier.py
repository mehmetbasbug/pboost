�
h*Qc           @   sx   d  d l  m Z d  d l m Z d  d l m Z m Z d  d l Z d �  Z e	 d k rt e j
 d Z e d e � n  d S(	   i����(   t   path(   t   loads(   t   signt   loadNc         C   sK  t  t |  � � } t  d � } d } xt | j d � D]� } | GHt j | | d d d � \ } } t j | � \ } } t | � }	 t |	 | | d d d � }
 |
 | � } t	 | | d d d � } | j
 | �  } | | | d d k r| | | d d 7} q8 | | | d d	 7} q8 Wd
 Gt t | � d � d GHd  S(   Ns   hypotheses.npyg        i    t   fn_defi   i   t   vt   c1t   c0s   Final classifier result =(   R   t   strt   ranget   shapeR    t   splitt   splitextt
   __import__t   getattrR   t   behaviort   intR   (   t   example_datat   datat
   hypothesest   resultt   kt   headt   tailt   roott   extt   mt	   bhv_classt   bhv_objt   argst   tmp(    (    s?   /home/mehmet/workspace/pboost/report/final_classifier_script.pyt   main   s     %t   __main__i   R   (   t   osR    t   jsonR   t   numpyR   R   t   sysR   t   __name__t   argvR   (    (    (    s?   /home/mehmet/workspace/pboost/report/final_classifier_script.pyt   <module>   s   	