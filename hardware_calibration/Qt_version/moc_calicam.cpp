/****************************************************************************
** Meta object code from reading C++ file 'calicam.h'
**
** Created by: The Qt Meta Object Compiler version 63 (Qt 4.8.6)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "calicam.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'calicam.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.6. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_caliCam[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
      11,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
       9,    8,    8,    8, 0x08,
      21,    8,    8,    8, 0x08,
      34,    8,    8,    8, 0x08,
      51,    8,    8,    8, 0x08,
      61,    8,    8,    8, 0x08,
      72,    8,    8,    8, 0x08,
      85,    8,    8,    8, 0x08,
     104,    8,    8,    8, 0x08,
     128,    8,    8,    8, 0x08,
     154,    8,    8,    8, 0x08,
     173,    8,    8,    8, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_caliCam[] = {
    "caliCam\0\0calibrate()\0saveImages()\0"
    "saveParameters()\0openCam()\0closeCam()\0"
    "camTimerUp()\0noSourceSelected()\0"
    "grabCamImg2ListWidget()\0"
    "removeImgFromListWidget()\0readImgFromFiles()\0"
    "cleanImageList()\0"
};

void caliCam::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        caliCam *_t = static_cast<caliCam *>(_o);
        switch (_id) {
        case 0: _t->calibrate(); break;
        case 1: _t->saveImages(); break;
        case 2: _t->saveParameters(); break;
        case 3: _t->openCam(); break;
        case 4: _t->closeCam(); break;
        case 5: _t->camTimerUp(); break;
        case 6: _t->noSourceSelected(); break;
        case 7: _t->grabCamImg2ListWidget(); break;
        case 8: _t->removeImgFromListWidget(); break;
        case 9: _t->readImgFromFiles(); break;
        case 10: _t->cleanImageList(); break;
        default: ;
        }
    }
    Q_UNUSED(_a);
}

const QMetaObjectExtraData caliCam::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject caliCam::staticMetaObject = {
    { &QWidget::staticMetaObject, qt_meta_stringdata_caliCam,
      qt_meta_data_caliCam, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &caliCam::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *caliCam::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *caliCam::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_caliCam))
        return static_cast<void*>(const_cast< caliCam*>(this));
    return QWidget::qt_metacast(_clname);
}

int caliCam::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 11)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 11;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
