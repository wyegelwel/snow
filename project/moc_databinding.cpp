/****************************************************************************
** Meta object code from reading C++ file 'databinding.h'
**
** Created: Tue Apr 29 20:16:27 2014
**      by: The Qt Meta Object Compiler version 63 (Qt 4.8.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "ui/databinding.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'databinding.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_IntBinding[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       2,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      18,   12,   11,   11, 0x0a,
      36,   12,   11,   11, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_IntBinding[] = {
    "IntBinding\0\0value\0valueChanged(int)\0"
    "valueChanged(QString)\0"
};

void IntBinding::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        IntBinding *_t = static_cast<IntBinding *>(_o);
        switch (_id) {
        case 0: _t->valueChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 1: _t->valueChanged((*reinterpret_cast< QString(*)>(_a[1]))); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData IntBinding::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject IntBinding::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_IntBinding,
      qt_meta_data_IntBinding, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &IntBinding::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *IntBinding::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *IntBinding::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_IntBinding))
        return static_cast<void*>(const_cast< IntBinding*>(this));
    return QObject::qt_metacast(_clname);
}

int IntBinding::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 2)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 2;
    }
    return _id;
}
static const uint qt_meta_data_FloatBinding[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       2,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      20,   14,   13,   13, 0x0a,
      41,   14,   13,   13, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_FloatBinding[] = {
    "FloatBinding\0\0value\0valueChanged(double)\0"
    "valueChanged(QString)\0"
};

void FloatBinding::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        FloatBinding *_t = static_cast<FloatBinding *>(_o);
        switch (_id) {
        case 0: _t->valueChanged((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 1: _t->valueChanged((*reinterpret_cast< QString(*)>(_a[1]))); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData FloatBinding::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject FloatBinding::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_FloatBinding,
      qt_meta_data_FloatBinding, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &FloatBinding::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *FloatBinding::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *FloatBinding::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_FloatBinding))
        return static_cast<void*>(const_cast< FloatBinding*>(this));
    return QObject::qt_metacast(_clname);
}

int FloatBinding::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 2)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 2;
    }
    return _id;
}
static const uint qt_meta_data_BoolBinding[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       1,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      19,   13,   12,   12, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_BoolBinding[] = {
    "BoolBinding\0\0value\0valueChanged(bool)\0"
};

void BoolBinding::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        BoolBinding *_t = static_cast<BoolBinding *>(_o);
        switch (_id) {
        case 0: _t->valueChanged((*reinterpret_cast< bool(*)>(_a[1]))); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData BoolBinding::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject BoolBinding::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_BoolBinding,
      qt_meta_data_BoolBinding, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &BoolBinding::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *BoolBinding::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *BoolBinding::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_BoolBinding))
        return static_cast<void*>(const_cast< BoolBinding*>(this));
    return QObject::qt_metacast(_clname);
}

int BoolBinding::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 1)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 1;
    }
    return _id;
}
static const uint qt_meta_data_SliderIntAttribute[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       3,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: signature, parameters, type, tag, flags
      26,   20,   19,   19, 0x05,

 // slots: signature, parameters, type, tag, flags
      48,   20,   19,   19, 0x0a,
      66,   20,   19,   19, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_SliderIntAttribute[] = {
    "SliderIntAttribute\0\0value\0"
    "attributeChanged(int)\0valueChanged(int)\0"
    "valueChanged(QString)\0"
};

void SliderIntAttribute::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        SliderIntAttribute *_t = static_cast<SliderIntAttribute *>(_o);
        switch (_id) {
        case 0: _t->attributeChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 1: _t->valueChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 2: _t->valueChanged((*reinterpret_cast< QString(*)>(_a[1]))); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData SliderIntAttribute::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject SliderIntAttribute::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_SliderIntAttribute,
      qt_meta_data_SliderIntAttribute, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &SliderIntAttribute::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *SliderIntAttribute::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *SliderIntAttribute::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_SliderIntAttribute))
        return static_cast<void*>(const_cast< SliderIntAttribute*>(this));
    return QObject::qt_metacast(_clname);
}

int SliderIntAttribute::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 3)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 3;
    }
    return _id;
}

// SIGNAL 0
void SliderIntAttribute::attributeChanged(int _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}
static const uint qt_meta_data_SliderFloatAttribute[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       4,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: signature, parameters, type, tag, flags
      28,   22,   21,   21, 0x05,

 // slots: signature, parameters, type, tag, flags
      52,   22,   21,   21, 0x0a,
      72,   22,   21,   21, 0x0a,
      90,   22,   21,   21, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_SliderFloatAttribute[] = {
    "SliderFloatAttribute\0\0value\0"
    "attributeChanged(float)\0valueChanged(float)\0"
    "valueChanged(int)\0valueChanged(QString)\0"
};

void SliderFloatAttribute::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        SliderFloatAttribute *_t = static_cast<SliderFloatAttribute *>(_o);
        switch (_id) {
        case 0: _t->attributeChanged((*reinterpret_cast< float(*)>(_a[1]))); break;
        case 1: _t->valueChanged((*reinterpret_cast< float(*)>(_a[1]))); break;
        case 2: _t->valueChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 3: _t->valueChanged((*reinterpret_cast< QString(*)>(_a[1]))); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData SliderFloatAttribute::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject SliderFloatAttribute::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_SliderFloatAttribute,
      qt_meta_data_SliderFloatAttribute, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &SliderFloatAttribute::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *SliderFloatAttribute::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *SliderFloatAttribute::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_SliderFloatAttribute))
        return static_cast<void*>(const_cast< SliderFloatAttribute*>(this));
    return QObject::qt_metacast(_clname);
}

int SliderFloatAttribute::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 4)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 4;
    }
    return _id;
}

// SIGNAL 0
void SliderFloatAttribute::attributeChanged(float _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}
static const uint qt_meta_data_CheckboxBoolAttribute[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       2,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: signature, parameters, type, tag, flags
      29,   23,   22,   22, 0x05,

 // slots: signature, parameters, type, tag, flags
      53,   23,   22,   22, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_CheckboxBoolAttribute[] = {
    "CheckboxBoolAttribute\0\0value\0"
    "attributedChanged(bool)\0valueChanged(bool)\0"
};

void CheckboxBoolAttribute::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        CheckboxBoolAttribute *_t = static_cast<CheckboxBoolAttribute *>(_o);
        switch (_id) {
        case 0: _t->attributedChanged((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 1: _t->valueChanged((*reinterpret_cast< bool(*)>(_a[1]))); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData CheckboxBoolAttribute::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject CheckboxBoolAttribute::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_CheckboxBoolAttribute,
      qt_meta_data_CheckboxBoolAttribute, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &CheckboxBoolAttribute::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *CheckboxBoolAttribute::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *CheckboxBoolAttribute::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_CheckboxBoolAttribute))
        return static_cast<void*>(const_cast< CheckboxBoolAttribute*>(this));
    return QObject::qt_metacast(_clname);
}

int CheckboxBoolAttribute::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 2)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 2;
    }
    return _id;
}

// SIGNAL 0
void CheckboxBoolAttribute::attributedChanged(bool _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}
static const uint qt_meta_data_ComboIntAttribute[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       2,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: signature, parameters, type, tag, flags
      25,   19,   18,   18, 0x05,

 // slots: signature, parameters, type, tag, flags
      47,   19,   18,   18, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_ComboIntAttribute[] = {
    "ComboIntAttribute\0\0value\0attributeChanged(int)\0"
    "valueChanged(int)\0"
};

void ComboIntAttribute::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        ComboIntAttribute *_t = static_cast<ComboIntAttribute *>(_o);
        switch (_id) {
        case 0: _t->attributeChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 1: _t->valueChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData ComboIntAttribute::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject ComboIntAttribute::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_ComboIntAttribute,
      qt_meta_data_ComboIntAttribute, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &ComboIntAttribute::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *ComboIntAttribute::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *ComboIntAttribute::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_ComboIntAttribute))
        return static_cast<void*>(const_cast< ComboIntAttribute*>(this));
    return QObject::qt_metacast(_clname);
}

int ComboIntAttribute::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 2)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 2;
    }
    return _id;
}

// SIGNAL 0
void ComboIntAttribute::attributeChanged(int _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}
QT_END_MOC_NAMESPACE
