#include "../core/ome_tiff_to_chunked_pyramid.h"
#include "../core/pyramid_view.h"
#include "../core/pyramid_compositor.h"
#include "../utilities/utilities.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

PYBIND11_MODULE(libargolid, m) {
    py::class_<argolid::OmeTiffToChunkedPyramid, std::shared_ptr<argolid::OmeTiffToChunkedPyramid>>(m, "OmeTiffToChunkedPyramidCPP") \
    .def(py::init<>()) \
    .def("GenerateFromSingleFile", &argolid::OmeTiffToChunkedPyramid::GenerateFromSingleFile) \
    .def("GenerateFromCollection", &argolid::OmeTiffToChunkedPyramid::GenerateFromCollection) \
    .def("SetLogLevel", &argolid::OmeTiffToChunkedPyramid::SetLogLevel) ;

    py::class_<argolid::PyramidView, std::shared_ptr<argolid::PyramidView>>(m, "PyramidViewCPP") \
    .def(py::init<std::string_view, std::string_view, std::string_view, std::uint16_t, std::uint16_t>()) \
    .def("GeneratePyramid", &argolid::PyramidView::GeneratePyramid) \
    .def("AssembleBaseLevel", &argolid::PyramidView::AssembleBaseLevel) ;


    py::class_<argolid::PyramidCompositor>(m, "PyramidCompositorCPP") \
    .def(py::init<const std::string&, const std::string&, const std::string&>()) \
    .def("reset_composition", &argolid::PyramidCompositor::reset_composition) \
    .def("create_xml", &argolid::PyramidCompositor::create_xml) \
    .def("create_zattr_file", &argolid::PyramidCompositor::create_zattr_file) \
    .def("create_auxiliary_files", &argolid::PyramidCompositor::create_auxiliary_files) \
    .def("write_zarr_chunk", &argolid::PyramidCompositor::write_zarr_chunk) \
    .def("set_composition", &argolid::PyramidCompositor::set_composition);

    py::enum_<argolid::VisType>(m, "VisType")
        .value("NG_Zarr", argolid::VisType::NG_Zarr)
        .value("PCNG", argolid::VisType::PCNG)
        .value("Viv", argolid::VisType::Viv)
        .export_values();

    py::enum_<argolid::DSType>(m, "DSType")
        .value("Mode_Max", argolid::DSType::Mode_Max)
        .value("Mode_Min", argolid::DSType::Mode_Min)
        .value("Mean", argolid::DSType::Mean)
        .export_values();
}