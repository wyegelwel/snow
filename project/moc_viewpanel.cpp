/****************************************************************************
** Meta object code from reading C++ file 'viewpanel.h'
**
** Created: Tue Apr 29 20:16:24 2014
**      by: The Qt Meta Object Compiler version 63 (Qt 4.8.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "ui/viewpanel.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'viewpanel.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_ViewPanel[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
      26,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       2,       // signalCount

 // signals: signature, parameters, type, tag, flags
      11,   10,   10,   10, 0x05,
      24,   10,   10,   10, 0x05,

 // slots: signature, parameters, type, tag, flags
      40,   10,   10,   10, 0x0a,
      56,   10,   10,   10, 0x0a,
      71,   10,   10,   10, 0x0a,
      87,   81,   10,   10, 0x0a,
     114,   81,   10,   10, 0x0a,
     144,   81,   10,   10, 0x0a,
     173,   81,   10,   10, 0x0a,
     205,   81,   10,   10, 0x0a,
     231,   10,   10,   10, 0x0a,
     255,  249,   10,   10, 0x0a,
     277,   10,   10,   10, 0x2a,
     295,   10,   10,   10, 0x0a,
     314,   10,   10,   10, 0x0a,
     329,   10,   10,   10, 0x0a,
     354,  345,   10,   10, 0x0a,
     384,  372,   10,   10, 0x0a,
     423,  418,   10,   10, 0x0a,
     436,   10,   10,   10, 0x0a,
     454,   10,   10,   10, 0x0a,
     471,   10,   10,   10, 0x0a,
     490,   10,   10,   10, 0x0a,
     514,   10,  509,   10, 0x0a,
     526,   10,  509,   10, 0x0a,
     538,   10,   10,   10, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_ViewPanel[] = {
    "ViewPanel\0\0showMeshes()\0showParticles()\0"
    "resetViewport()\0initializeGL()\0paintGL()\0"
    "event\0resizeEvent(QResizeEvent*)\0"
    "mousePressEvent(QMouseEvent*)\0"
    "mouseMoveEvent(QMouseEvent*)\0"
    "mouseReleaseEvent(QMouseEvent*)\0"
    "keyPressEvent(QKeyEvent*)\0resetSimulation()\0"
    "pause\0pauseSimulation(bool)\0"
    "pauseSimulation()\0resumeSimulation()\0"
    "pauseDrawing()\0resumeDrawing()\0filename\0"
    "loadMesh(QString)\0c,planeType\0"
    "addCollider(ColliderType,QString)\0"
    "tool\0setTool(int)\0updateSceneGrid()\0"
    "clearSelection()\0fillSelectedMesh()\0"
    "saveSelectedMesh()\0bool\0loadScene()\0"
    "saveScene()\0teapotDemo()\0"
};

void ViewPanel::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        ViewPanel *_t = static_cast<ViewPanel *>(_o);
        switch (_id) {
        case 0: _t->showMeshes(); break;
        case 1: _t->showParticles(); break;
        case 2: _t->resetViewport(); break;
        case 3: _t->initializeGL(); break;
        case 4: _t->paintGL(); break;
        case 5: _t->resizeEvent((*reinterpret_cast< QResizeEvent*(*)>(_a[1]))); break;
        case 6: _t->mousePressEvent((*reinterpret_cast< QMouseEvent*(*)>(_a[1]))); break;
        case 7: _t->mouseMoveEvent((*reinterpret_cast< QMouseEvent*(*)>(_a[1]))); break;
        case 8: _t->mouseReleaseEvent((*reinterpret_cast< QMouseEvent*(*)>(_a[1]))); break;
        case 9: _t->keyPressEvent((*reinterpret_cast< QKeyEvent*(*)>(_a[1]))); break;
        case 10: _t->resetSimulation(); break;
        case 11: _t->pauseSimulation((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 12: _t->pauseSimulation(); break;
        case 13: _t->resumeSimulation(); break;
        case 14: _t->pauseDrawing(); break;
        case 15: _t->resumeDrawing(); break;
        case 16: _t->loadMesh((*reinterpret_cast< const QString(*)>(_a[1]))); break;
        case 17: _t->addCollider((*reinterpret_cast< ColliderType(*)>(_a[1])),(*reinterpret_cast< QString(*)>(_a[2]))); break;
        case 18: _t->setTool((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 19: _t->updateSceneGrid(); break;
        case 20: _t->clearSelection(); break;
        case 21: _t->fillSelectedMesh(); break;
        case 22: _t->saveSelectedMesh(); break;
        case 23: { bool _r = _t->loadScene();
            if (_a[0]) *reinterpret_cast< bool*>(_a[0]) = _r; }  break;
        case 24: { bool _r = _t->saveScene();
            if (_a[0]) *reinterpret_cast< bool*>(_a[0]) = _r; }  break;
        case 25: _t->teapotDemo(); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData ViewPanel::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject ViewPanel::staticMetaObject = {
    { &QGLWidget::staticMetaObject, qt_meta_stringdata_ViewPanel,
      qt_meta_data_ViewPanel, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &ViewPanel::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *ViewPanel::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *ViewPanel::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_ViewPanel))
        return static_cast<void*>(const_cast< ViewPanel*>(this));
    return QGLWidget::qt_metacast(_clname);
}

int ViewPanel::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QGLWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 26)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 26;
    }
    return _id;
}

// SIGNAL 0
void ViewPanel::showMeshes()
{
    QMetaObject::activate(this, &staticMetaObject, 0, 0);
}

// SIGNAL 1
void ViewPanel::showParticles()
{
    QMetaObject::activate(this, &staticMetaObject, 1, 0);
}
QT_END_MOC_NAMESPACE
