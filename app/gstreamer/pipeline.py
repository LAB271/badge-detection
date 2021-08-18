#! /usr/bin/env python
import logging

import gi
from gi.repository import GObject

gi.require_version('Gst', '1.0')
from gi.repository import GLib
from gi.repository import Gst

log = logging.getLogger(__name__)

Gst.init_check(None)

LOOP = GObject.MainLoop()


class ComponentNamespace(object):
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def __getattribute__(self, name):
        if name != 'pipeline':
            return self.pipeline.get_child_by_name(name)
        return super(ComponentNamespace, self).__getattribute__(name)


class Pipe(object):
    def __init__(self, name, rtsp_url):
        self.name = name

        # This one is faster, but the size of the image is always 1.5 times the size of the buffer, and therefore it cannot be read.
        # self.pipeline_command = ("rtspsrc location={} latency=0 ! queue ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! videoscale ! video/x-raw ! appsink name=app_sink drop=false max-buffers=2").format(rtsp_url)
        #self.pipeline_command = ("rtspsrc location={} latency=0 ! queue ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! videorate ! video/x-raw,framerate=5/1 ! jpegenc quality=75 ! appsink name=app_sink drop=true max-buffers=10").format(rtsp_url)
        self.pipeline_command = ("rtspsrc location={} latency=0 protocols=tcp name=rtsp_src tcp-timeout=10000000 ! decodebin ! videoconvert ! videorate ! video/x-raw,framerate=5/1 ! jpegenc quality=85 ! appsink name=app_sink drop=false max-buffers=2").format(rtsp_url)
        self.pipeline = None
        self.state = Gst.State.NULL
        self.appsink = None

    def run(self):
        try:
            self.pipeline = Gst.parse_launch(self.pipeline_command)
        except GLib.Error as err:
            log.error("Failed to parse the pipeline: %s", err)
            log.info(
                " gst-launch --gst-debug=3 %s",
                " ".join(self.pipeline_command)
            )
            raise
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.message)
        self.appsink = self.pipeline.get_by_name("app_sink")
        # Set properties
        self.appsink.set_property('emit-signals', True)
        # turns off sync to make decoding as fast as possible
        self.appsink.set_property('sync', False)
        self.pipeline.set_state(Gst.State.PLAYING)

    def message(self, bus, message):
        """Process a bus message from gstreamer"""
        message_name = message.type.get_name(message.type).replace('-', '_')
        method_name = 'message_%s' % (message_name,)
        method = getattr(self, method_name, None)
        if method:
            return method(bus, message)
        else:
            log.info("message: %s", message.type)

    @property
    def components(self):
        return ComponentNamespace(self.pipeline)

    def get_appsink(self):
        return self.appsink

    def message_error(self, bus, message):
        err, debug = message.parse_error()
        log.error("Error reported, aborting: %s (debug=%s)", err, debug)
        LOOP.quit()

    def message_state_changed(self, bus, message):
        structure = message.parse_state_changed()
        newstate = structure.newstate
        if newstate != self.state:
            log.info('%s state: %s -> %s', self.name, self.state, newstate)
            self.state = newstate

    def message_stream_start(self, bus, message):
        log.info("Stream started")

    def message_stream_status(self, bus, message):
        log.info('Stream status: %s', message.parse_stream_status())
        self.log_pad_structures()

    def log_pad_structures(self):
        if self.components.demux:
            for i, pad in enumerate(self.components.demux.srcpads):
                log.info("Demux pad: %s", pad.name)
        elif self.components.muxer:
            for i, pad in enumerate(self.components.muxer.sinkpads):
                log.info("Muxer pad: %s", pad.name)
