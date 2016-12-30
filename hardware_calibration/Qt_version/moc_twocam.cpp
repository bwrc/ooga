/****************************************************************************
** Meta object code from reading C++ file 'twocam.h'
**
** Created by: The Qt Meta Object Compiler version 63 (Qt 4.8.6)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "twocam.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'twocam.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.6. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_TwoCam[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
      10,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
       8,    7,    7,    7, 0x08,
      21,    7,    7,    7, 0x08,
      31,    7,    7,    7, 0x08,
      42,    7,    7,    7, 0x08,
      55,    7,    7,    7, 0x08,
      74,    7,    7,    7, 0x08,
     100,    7,    7,    7, 0x08,
     131,    7,    7,    7, 0x08,
     164,    7,    7,    7, 0x08,
     181,    7,    7,    7, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_TwoCam[] = {
    "TwoCam\0\0saveImages()\0openCam()\0"
    "closeCam()\0camTimerUp()\0noSourceSelected()\0"
    "grabCamImage2ListWidget()\0"
    "removeEyeImageFromListWidget()\0"
    "removeSceneImageFromListWidget()\0"
    "readImageFiles()\0cleanImageList()\0"
};

void TwoCam::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        TwoCam *_t = static_cast<TwoCam *>(_o);
        switch (_id) {
        case 0: _t->saveImages(); break;
        case 1: _t->openCam(); break;
        case 2: _t->closeCam(); break;
        case 3: _t->camTimerUp(); break;
        case 4: _t->noSourceSelected(); break;
        case 5: _t->grabCamImage2ListWidget(); break;
        case 6: _t->removeEyeImageFromListWidget(); break;
        case 7: _t->removeSceneImageFromListWidget(); break;
        case 8: _t->readImageFiles(); break;
        case 9: _t->cleanImageList(); break;
        default: ;
        }
    }
    Q_UNUSED(_a);
}

const QMetaObjectExtraData TwoCam::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject TwoCam::staticMetaObject = {
    { &QWidget::staticMetaObject, qt_meta_stringdata_TwoCam,
      qt_meta_data_TwoCam, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &TwoCam::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *TwoCam::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *TwoCam::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_TwoCam))
        return static_cast<void*>(const_cast< TwoCam*>(this));
    return QWidget::qt_metacast(_clname);
}

int TwoCam::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 10)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 10;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
