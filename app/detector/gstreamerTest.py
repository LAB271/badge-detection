import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
from gi.repository import GObject, Gst, GLib, GstApp
GObject.threads_init()
Gst.init(None)
URI = "rtsp://kplr:5hVUlm3S7o92@10.10.12.241/Streaming/channels/101"
mainloop = GObject.MainLoop()
pipeline_string = pipeline = ("rtspsrc location={} protocols=tcp name=rtsp_src tcp-timeout=10000000 ! decodebin ! " +
                        "videoconvert ! videorate ! video/x-raw,framerate=1/1 ! jpegenc quality=95 ! " +
                        "appsink name=app_sink drop=true max-buffers=10").format(URI)
pipeline = Gst.parse_launch(pipeline_string)
bus = pipeline.get_bus()
bus.add_signal_watch()
sink = pipeline.get_by_name("app_sink")
# Set properties
sink.set_property('emit-signals', True)
# turns off sync to make decoding as fast as possible
sink.set_property('sync', False)
pipeline.set_state(Gst.State.PLAYING)
print("-----")
mainloop.run()

sample = sink.try_pull_sample(Gst.SECOND)

appsink_sample = GstApp.AppSink.pull_sample(sink)
if appsink_sample is None:
    print("appsink sample is None, terminating")
