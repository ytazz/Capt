/****************************************************************************
** Meta object code from reading C++ file 'setting_item.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.12.3)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../window/include/setting_item.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'setting_item.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.12.3. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_CA__SettingItem_t {
    QByteArrayData data[19];
    char stringdata0[183];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_CA__SettingItem_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_CA__SettingItem_t qt_meta_stringdata_CA__SettingItem = {
    {
QT_MOC_LITERAL(0, 0, 15), // "CA::SettingItem"
QT_MOC_LITERAL(1, 16, 18), // "setPolarGridRadius"
QT_MOC_LITERAL(2, 35, 0), // ""
QT_MOC_LITERAL(3, 36, 3), // "min"
QT_MOC_LITERAL(4, 40, 3), // "max"
QT_MOC_LITERAL(5, 44, 4), // "step"
QT_MOC_LITERAL(6, 49, 11), // "const char*"
QT_MOC_LITERAL(7, 61, 10), // "color_name"
QT_MOC_LITERAL(8, 72, 17), // "setPolarGridAngle"
QT_MOC_LITERAL(9, 90, 8), // "setPoint"
QT_MOC_LITERAL(10, 99, 7), // "Vector2"
QT_MOC_LITERAL(11, 107, 5), // "point"
QT_MOC_LITERAL(12, 113, 9), // "setPoints"
QT_MOC_LITERAL(13, 123, 20), // "std::vector<Vector2>"
QT_MOC_LITERAL(14, 144, 10), // "setPolygon"
QT_MOC_LITERAL(15, 155, 6), // "vertex"
QT_MOC_LITERAL(16, 162, 5), // "paint"
QT_MOC_LITERAL(17, 168, 5), // "reset"
QT_MOC_LITERAL(18, 174, 8) // "openFile"

    },
    "CA::SettingItem\0setPolarGridRadius\0\0"
    "min\0max\0step\0const char*\0color_name\0"
    "setPolarGridAngle\0setPoint\0Vector2\0"
    "point\0setPoints\0std::vector<Vector2>\0"
    "setPolygon\0vertex\0paint\0reset\0openFile"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_CA__SettingItem[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       8,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       7,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    4,   54,    2, 0x06 /* Public */,
       8,    4,   63,    2, 0x06 /* Public */,
       9,    2,   72,    2, 0x06 /* Public */,
      12,    2,   77,    2, 0x06 /* Public */,
      14,    2,   82,    2, 0x06 /* Public */,
      16,    0,   87,    2, 0x06 /* Public */,
      17,    0,   88,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
      18,    0,   89,    2, 0x0a /* Public */,

 // signals: parameters
    QMetaType::Void, QMetaType::Float, QMetaType::Float, QMetaType::Float, 0x80000000 | 6,    3,    4,    5,    7,
    QMetaType::Void, QMetaType::Float, QMetaType::Float, QMetaType::Float, 0x80000000 | 6,    3,    4,    5,    7,
    QMetaType::Void, 0x80000000 | 10, 0x80000000 | 6,   11,    7,
    QMetaType::Void, 0x80000000 | 13, 0x80000000 | 6,   11,    7,
    QMetaType::Void, 0x80000000 | 13, 0x80000000 | 6,   15,    7,
    QMetaType::Void,
    QMetaType::Void,

 // slots: parameters
    QMetaType::Void,

       0        // eod
};

void CA::SettingItem::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<SettingItem *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->setPolarGridRadius((*reinterpret_cast< float(*)>(_a[1])),(*reinterpret_cast< float(*)>(_a[2])),(*reinterpret_cast< float(*)>(_a[3])),(*reinterpret_cast< const char*(*)>(_a[4]))); break;
        case 1: _t->setPolarGridAngle((*reinterpret_cast< float(*)>(_a[1])),(*reinterpret_cast< float(*)>(_a[2])),(*reinterpret_cast< float(*)>(_a[3])),(*reinterpret_cast< const char*(*)>(_a[4]))); break;
        case 2: _t->setPoint((*reinterpret_cast< Vector2(*)>(_a[1])),(*reinterpret_cast< const char*(*)>(_a[2]))); break;
        case 3: _t->setPoints((*reinterpret_cast< std::vector<Vector2>(*)>(_a[1])),(*reinterpret_cast< const char*(*)>(_a[2]))); break;
        case 4: _t->setPolygon((*reinterpret_cast< std::vector<Vector2>(*)>(_a[1])),(*reinterpret_cast< const char*(*)>(_a[2]))); break;
        case 5: _t->paint(); break;
        case 6: _t->reset(); break;
        case 7: _t->openFile(); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (SettingItem::*)(float , float , float , const char * );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&SettingItem::setPolarGridRadius)) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (SettingItem::*)(float , float , float , const char * );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&SettingItem::setPolarGridAngle)) {
                *result = 1;
                return;
            }
        }
        {
            using _t = void (SettingItem::*)(Vector2 , const char * );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&SettingItem::setPoint)) {
                *result = 2;
                return;
            }
        }
        {
            using _t = void (SettingItem::*)(std::vector<Vector2> , const char * );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&SettingItem::setPoints)) {
                *result = 3;
                return;
            }
        }
        {
            using _t = void (SettingItem::*)(std::vector<Vector2> , const char * );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&SettingItem::setPolygon)) {
                *result = 4;
                return;
            }
        }
        {
            using _t = void (SettingItem::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&SettingItem::paint)) {
                *result = 5;
                return;
            }
        }
        {
            using _t = void (SettingItem::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&SettingItem::reset)) {
                *result = 6;
                return;
            }
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject CA::SettingItem::staticMetaObject = { {
    &QWidget::staticMetaObject,
    qt_meta_stringdata_CA__SettingItem.data,
    qt_meta_data_CA__SettingItem,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *CA::SettingItem::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *CA::SettingItem::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_CA__SettingItem.stringdata0))
        return static_cast<void*>(this);
    return QWidget::qt_metacast(_clname);
}

int CA::SettingItem::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 8)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 8;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 8)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 8;
    }
    return _id;
}

// SIGNAL 0
void CA::SettingItem::setPolarGridRadius(float _t1, float _t2, float _t3, const char * _t4)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)), const_cast<void*>(reinterpret_cast<const void*>(&_t3)), const_cast<void*>(reinterpret_cast<const void*>(&_t4)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void CA::SettingItem::setPolarGridAngle(float _t1, float _t2, float _t3, const char * _t4)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)), const_cast<void*>(reinterpret_cast<const void*>(&_t3)), const_cast<void*>(reinterpret_cast<const void*>(&_t4)) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}

// SIGNAL 2
void CA::SettingItem::setPoint(Vector2 _t1, const char * _t2)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 2, _a);
}

// SIGNAL 3
void CA::SettingItem::setPoints(std::vector<Vector2> _t1, const char * _t2)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 3, _a);
}

// SIGNAL 4
void CA::SettingItem::setPolygon(std::vector<Vector2> _t1, const char * _t2)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 4, _a);
}

// SIGNAL 5
void CA::SettingItem::paint()
{
    QMetaObject::activate(this, &staticMetaObject, 5, nullptr);
}

// SIGNAL 6
void CA::SettingItem::reset()
{
    QMetaObject::activate(this, &staticMetaObject, 6, nullptr);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
