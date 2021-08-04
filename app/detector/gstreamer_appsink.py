import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
from gi.repository import GObject, Gst, GLib, GstApp
import threading
import time

GObject.threads_init()
Gst.init(None)

command = "videotestsrc num-buffers=100 ! \
    capsfilter caps=video/x-raw,format=RGB,width=640,height=480 ! \
    appsink emit-signals=True"

class GstContext:
    def __init__(self):
        # SIGINT handle issue:
        # https://github.com/beetbox/audioread/issues/63#issuecomment-390394735
        self._main_loop = GLib.MainLoop.new(None, False)

        self._main_loop_thread = threading.Thread(target=self._main_loop_run)

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return "<{}>".format(self)

    def __enter__(self):
        self.startup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def startup(self):
        if self._main_loop_thread.is_alive():
            return

        self._main_loop_thread.start()

    def _main_loop_run(self):
        try:
            self._main_loop.run()
        except Exception:
            pass

    def shutdown(self, timeout: int = 2):
        #self.log.debug("%s Quitting main loop ...", self)

        if self._main_loop.is_running():
            self._main_loop.quit()

        #self.log.debug("%s Joining main loop thread...", self)
        try:
            if self._main_loop_thread.is_alive():
                self._main_loop_thread.join(timeout=timeout)
        except Exception as err:
            #self.log.error("%s.main_loop_thread : %s", self, err)
            pass


class GstPipeline:
    """Base class to initialize any Gstreamer Pipeline from string"""

    def __init__(self, command: str):
        """
        :param command: gst-launch string
        """
        self._command = command
        self._pipeline = None  # Gst.Pipeline
        self._bus = None  # Gst.Bus

        #self._log.info("%s \n gst-launch-1.0 %s", self, command)

        self._end_stream_event = threading.Event()

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return "<{}>".format(self)

    def __enter__(self):
        self.startup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def get_by_cls(self, cls: GObject.GType):
        """ Get Gst.Element[] from pipeline by GType """
        elements = self._pipeline.iterate_elements()
        if isinstance(elements, Gst.Iterator):
            # Patch "TypeError: ‘Iterator’ object is not iterable."
            # For versions we have to get a python iterable object from Gst iterator
            _elements = []
            while True:
                ret, el = elements.next()
                if ret == Gst.IteratorResult(1):  # GST_ITERATOR_OK
                    _elements.append(el)
                else:
                    break
            elements = _elements

        return [e for e in elements if isinstance(e, cls)]

    def get_by_name(self, name: str) -> Gst.Element:
        """Get Gst.Element from pipeline by name
        :param name: plugins name (name={} in gst-launch string)
        """
        return self._pipeline.get_by_name(name)

    def startup(self):
        """ Starts pipeline """
        if self._pipeline:
            raise RuntimeError("Can't initiate %s. Already started")

        self._pipeline = Gst.parse_launch(self._command)

        # Initialize Bus
        self._bus = self._pipeline.get_bus()
        self._bus.add_signal_watch()
        self.bus.connect("message::error", self.on_error)
        self.bus.connect("message::eos", self.on_eos)
        self.bus.connect("message::warning", self.on_warning)

        # Initalize Pipeline
        self._on_pipeline_init()
        self._pipeline.set_state(Gst.State.READY)

        #self.log.info("Starting %s", self)

        self._end_stream_event.clear()

        self._pipeline.set_state(Gst.State.PLAYING)
        #self.log.debug("%s Pipeline state set to %s ", self, Gst.Element.state_get_name(Gst.State.PLAYING))

    def _on_pipeline_init(self) -> None:
        """Sets additional properties for plugins in Pipeline"""
        pass

    @property
    def bus(self) -> Gst.Bus:
        return self._bus

    @property
    def pipeline(self) -> Gst.Pipeline:
        return self._pipeline

    def _shutdown_pipeline(self, timeout: int = 1, eos: bool = False) -> None:
        """ Stops pipeline
        :param eos: if True -> send EOS event
            - EOS event necessary for FILESINK finishes properly
            - Use when pipeline crushes
        """

        if self._end_stream_event.is_set():
            return

        self._end_stream_event.set()

        if not self.pipeline:
            return

        #self.log.debug("%s Stopping pipeline ...", self)

        # https://lazka.github.io/pgi-docs/Gst-1.0/classes/Element.html#Gst.Element.get_state
        if self._pipeline.get_state(timeout=1)[1] == Gst.State.PLAYING:
            #self.log.debug("%s Sending EOS event ...", self)
            try:
                thread = threading.Thread(
                    target=self._pipeline.send_event, args=(Gst.Event.new_eos(),)
                )
                thread.start()
                thread.join(timeout=timeout)
            except Exception:
                pass

        #self.log.debug("%s Reseting pipeline state ....", self)
        try:
            self._pipeline.set_state(Gst.State.NULL)
            self._pipeline = None
        except Exception:
            pass

        #self.log.debug("%s Gst.Pipeline successfully destroyed", self)

    def shutdown(self, timeout: int = 1, eos: bool = False) -> None:
        """Shutdown pipeline
        :param timeout: time to wait when pipeline fully stops
        :param eos: if True -> send EOS event
            - EOS event necessary for FILESINK finishes properly
            - Use when pipeline crushes
        """
        #self.log.info("%s Shutdown requested ...", self)

        self._shutdown_pipeline(timeout=timeout, eos=eos)

        #self.log.info("%s successfully destroyed", self)

    @property
    def is_active(self) -> bool:
        return self.pipeline is not None and not self.is_done

    @property
    def is_done(self) -> bool:
        return self._end_stream_event.is_set()

    def on_error(self, bus: Gst.Bus, message: Gst.Message):
        err, debug = message.parse_error()
        #self.log.error("Gstreamer.%s: Error %s: %s. ", self, err, debug)
        self._shutdown_pipeline()

    def on_eos(self, bus: Gst.Bus, message: Gst.Message):
        #self.log.debug("Gstreamer.%s: Received stream EOS event", self)
        self._shutdown_pipeline()

    def on_warning(self, bus: Gst.Bus, message: Gst.Message):
        warn, debug = message.parse_warning()
        #self.log.warning("Gstreamer.%s: %s. %s", self, warn, debug)


def on_buffer(sink, data):
     sample = sink.emit("pull-sample")  # Gst.Sample
     if isinstance(sample, Gst.Sample):
        buffer = sample.get_buffer()  # Gst.Buffer
        print(buffer)

with GstContext():
    with GstPipeline(command) as pipeline:
        appsink = pipeline.get_by_cls(GstApp.AppSink).pop(0)
        appsink.connect("new-sample", on_buffer, None)
        while not pipeline.is_done:
            time.sleep(.1)