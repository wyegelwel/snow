/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   databinding.h
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 17 Apr 2014
**
**************************************************************************/

#ifndef DATABINDING_H
#define DATABINDING_H

#include "common/common.h"

#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QLineEdit>
#include <QObject>
#include <QSlider>
#include <QSpinBox>

class IntBinding : public QObject
{

    Q_OBJECT

public:

    IntBinding( int &value, QObject *parent = NULL ) : QObject(parent), m_value(value) {}

    static IntBinding* bindSpinBox( QSpinBox *spinbox, int &value, QObject *parent = NULL )
    {
        IntBinding *binding = new IntBinding( value, parent );
        spinbox->setValue( value );
        assert( connect(spinbox, SIGNAL(valueChanged(int)), binding, SLOT(valueChanged(int))) );
        return binding;
    }

    static IntBinding* bindLineEdit( QLineEdit *lineEdit, int &value, QObject *parent = NULL )
    {
        IntBinding *binding = new IntBinding( value, parent );
        lineEdit->setText( QString::number(value) );
        assert( connect(lineEdit, SIGNAL(textChanged(QString)), binding, SLOT(valueChanged(QString))) );
        return binding;
    }

    static IntBinding* bindSlider( QSlider *slider, int &value, QObject *parent = NULL )
    {
        IntBinding *binding = new IntBinding( value, parent );
        slider->setValue( value );
        assert( connect(slider, SIGNAL(valueChanged(int)), binding, SLOT(valueChanged(int))) );
        return binding;
    }

    static IntBinding* bindTriState( QCheckBox *checkbox, int &value, QObject *parent = NULL )
    {
        IntBinding *binding = new IntBinding( value, parent );
        checkbox->setTristate( true );
        checkbox->setCheckState( (Qt::CheckState)(value) );
        assert( connect(checkbox, SIGNAL(stateChanged(int)), binding, SLOT(valueChanged(int))) );
        return binding;
    }

public slots:

    void valueChanged( int value )
    {
        m_value = value;
    }

    void valueChanged( QString value )
    {
        bool ok = false;
        int intValue = value.toInt( &ok );
        if ( ok ) m_value = intValue;
    }

private:

    int &m_value;

};

class FloatBinding : public QObject
{

    Q_OBJECT

public:

    FloatBinding( float &value, QObject *parent = NULL ) : QObject(parent), m_value(value) {}

    static FloatBinding* bindSpinBox( QDoubleSpinBox *spinbox, float &value, QObject *parent = NULL )
    {
        FloatBinding *binding = new FloatBinding( value, parent );
        spinbox->setValue( value );
        assert( connect(spinbox, SIGNAL(valueChanged(double)), binding, SLOT(valueChanged(double))) );
        return binding;
    }

    static FloatBinding* bindLineEdit( QLineEdit *lineEdit, float &value, QObject *parent = NULL )
    {
        FloatBinding *binding = new FloatBinding( value, parent );
        lineEdit->setText( QString::number(value) );
        assert( connect(lineEdit, SIGNAL(textChanged(QString)), binding, SLOT(valueChanged(QString))) );
        return binding;
    }

public slots:

    void valueChanged( double value )
    {
        m_value = (float)value;
    }

    void valueChanged( QString value )
    {
        bool ok = false;
        float floatValue = value.toFloat( &ok );
        if ( ok ) m_value = floatValue;
    }

private:

    float &m_value;

};

class BoolBinding : public QObject
{

    Q_OBJECT

public:

    BoolBinding( bool &value, QObject *parent = NULL ) : QObject(parent), m_value(value) {}

    static BoolBinding* bindCheckBox( QCheckBox *checkbox, bool &value, QObject *parent = NULL )
    {
        BoolBinding *binding = new BoolBinding( value, parent );
        checkbox->setChecked( value );
        assert( connect(checkbox, SIGNAL(toggled(bool)), binding, SLOT(valueChanged(bool))) );
        return binding;
    }

public slots:

    void valueChanged( bool value )
    {
        m_value = value;
    }

private:

    bool &m_value;
};

class SliderIntAttribute : public QObject
{

    Q_OBJECT

public:

    SliderIntAttribute( QSlider *slider, QLineEdit *edit,
                        int min, int max, int *value,
                        QObject *parent )
        : QObject(parent),
          m_slider(slider),
          m_edit(edit),
          m_value(value)
    {
        if ( m_value ) {
            m_slider->setValue( *m_value );
            m_edit->setText( QString::number(*m_value) );
        }
        m_slider->setMinimum( min );
        m_slider->setMaximum( max );
        m_edit->setValidator( new QIntValidator(min, max, m_edit) );
        assert( connect(m_slider, SIGNAL(valueChanged(int)), this, SLOT(valueChanged(int))) );
        assert( connect(m_edit, SIGNAL(textChanged(QString)), this, SLOT(valueChanged(QString))) );
    }

    static SliderIntAttribute* bindInt( QSlider *slider, QLineEdit *edit, int min, int max, int *value, QObject *parent )
    {
        return new SliderIntAttribute( slider, edit, min, max, value, parent );
    }

    static SliderIntAttribute* bindSlot( QSlider *slider, QLineEdit *edit, int min, int max, QObject *object, const char *slot )
    {
        SliderIntAttribute *attr = new SliderIntAttribute( slider, edit, min, max, NULL, object );
        assert( connect(attr, SIGNAL(attributeChanged(int)), object, slot) );
        return attr;
    }

    static SliderIntAttribute* bindIntAndSlot( QSlider *slider, QLineEdit *edit, int min, int max, int *value, QObject *object, const char *slot )
    {
        SliderIntAttribute *attr = new SliderIntAttribute( slider, edit, min, max, value, object );
        assert( connect(attr, SIGNAL(attributeChanged(int)), object, slot) );
        return attr;
    }

signals:

    void attributeChanged( int value );

public slots:

    void valueChanged( int value )
    {
        bool shouldEmit = true;
        if ( m_value ) {
            if ( (shouldEmit = (*m_value != value)) ) {
                *m_value = value;
            }
        }
        m_slider->setValue( value );
        m_edit->setText( QString::number(value) );
        if ( shouldEmit ) emit attributeChanged( value );
    }

    void valueChanged( QString value ) { valueChanged(value.toInt()); }

private:

    QSlider *m_slider;
    QLineEdit *m_edit;

    int *m_value;

};

class SliderFloatAttribute : public QObject
{

    Q_OBJECT

public:

    SliderFloatAttribute( QSlider *slider, QLineEdit *edit,
                          float min, float max, float *value,
                          QObject *parent )
        : QObject(parent),
          m_slider(slider),
          m_edit(edit),
          m_value(value),
          m_min(min),
          m_max(max)
    {
        if ( m_value ) {
            m_slider->setValue( intValue(*m_value) );
            m_edit->setText( QString::number(*m_value, 'f', 3) );
        }
        m_slider->setMinimum(0);
        m_slider->setMaximum( int(1000*(m_max-m_min)+0.5f) );
        m_edit->setValidator( new QDoubleValidator(m_min, m_max, 3, m_edit) );
        assert( connect(m_slider, SIGNAL(valueChanged(int)), this, SLOT(valueChanged(int))) );
        assert( connect(m_edit, SIGNAL(textChanged(QString)), this, SLOT(valueChanged(QString))) );
    }

    static SliderFloatAttribute* bindFloat( QSlider *slider, QLineEdit *edit, float min, float max, float *value, QObject *parent )
    {
        return new SliderFloatAttribute( slider, edit, min, max, value, parent );
    }

    static SliderFloatAttribute* bindSlot( QSlider *slider, QLineEdit *edit, float min, float max, QObject *object, const char *slot )
    {
        SliderFloatAttribute* attr = new SliderFloatAttribute( slider, edit, min, max, NULL, object );
        assert( connect(attr, SIGNAL(attributeChanged(float)), object, slot) );
        return attr;
    }

    static SliderFloatAttribute* bindFloatAndSlot( QSlider *slider, QLineEdit *edit, float min, float max, float *value, QObject *object, const char *slot )
    {
        SliderFloatAttribute* attr = new SliderFloatAttribute( slider, edit, min, max, value, object );
        assert( connect(attr, SIGNAL(attributeChanged(float)), object, slot ) );
        return attr;
    }

signals:

    void attributeChanged( float value );

public slots:

    void valueChanged( float value )
    {

        bool shouldEmit = true;
        if ( m_value ) {
            if ( (shouldEmit = (*m_value != value)) ) {
                *m_value = value;
            }
        }
        m_slider->setValue( intValue(value) );
        m_edit->setText( QString::number(value, 'f', 3) );
        emit attributeChanged( value );
    }

    void valueChanged( int value ) { valueChanged( floatValue(value) ); }

    void valueChanged( QString value ) { valueChanged(value.toFloat()); }

private:

    QSlider *m_slider;
    QLineEdit *m_edit;

    float *m_value;
    float m_min, m_max;

    inline float floatValue( int i )
    {
        float t = float( i - m_slider->minimum() ) / float( m_slider->maximum() - m_slider->minimum() );
        return m_min + t*(m_max-m_min);
    }

    inline int intValue( float f )
    {
        float t = ( f - m_min ) / ( m_max - m_min );
        return (int)( m_slider->minimum() + t*(m_slider->maximum()-m_slider->minimum()) + 0.5f );
    }

};

class CheckboxBoolAttribute : public QObject
{

    Q_OBJECT

public:

    CheckboxBoolAttribute( QCheckBox *checkbox, bool *value, QObject *parent )
        : QObject(parent),
          m_checkbox(checkbox),
          m_value(value)
    {
        if ( value ) {
            m_checkbox->setChecked( !(*m_value) );
            m_checkbox->click();
        }
        assert( connect(checkbox, SIGNAL(clicked(bool)), this, SLOT(valueChanged(bool))) );
    }

    static CheckboxBoolAttribute* bindBool( QCheckBox *checkbox, bool *value, QObject *parent )
    {
        return new CheckboxBoolAttribute( checkbox, value, parent );
    }

    static CheckboxBoolAttribute* bindSlot( QCheckBox *checkbox, QObject *object, const char *slot )
    {
        CheckboxBoolAttribute *attr = new CheckboxBoolAttribute( checkbox, NULL, object );
        assert( connect(attr, SIGNAL(attributedChanged(bool)), object, slot) );
        return attr;
    }

    static CheckboxBoolAttribute* bindBoolAndSlot( QCheckBox *checkbox, bool *value, QObject *object, const char *slot )
    {
        CheckboxBoolAttribute *attr = new CheckboxBoolAttribute( checkbox, value, object );
        assert( connect(attr, SIGNAL(attributedChanged(bool)), object, slot) );
        return attr;
    }

signals:

    void attributedChanged( bool value );

public slots:

    void valueChanged( bool value )
    {
        bool shouldEmit = true;
        if ( m_value ) {
            if ( (shouldEmit = (*m_value != value)) ) {
                *m_value = value;
            }
        }
        m_checkbox->setChecked( value );
        if ( shouldEmit ) emit attributedChanged( value );
    }

private:

    QCheckBox *m_checkbox;

    bool *m_value;

};

class ComboIntAttribute : public QObject
{

    Q_OBJECT

public:

    ComboIntAttribute( QComboBox *combo, int *value, QObject *parent )
        : QObject(parent),
          m_combo(combo),
          m_value(value)
    {
        if ( m_value ) {
            m_combo->setCurrentIndex( *m_value );
        }
        assert( connect(m_combo, SIGNAL(currentIndexChanged(int)), this, SLOT(valueChanged(int))) );
    }

    static ComboIntAttribute* bindInt( QComboBox *combo, int *value, QObject *parent )
    {
        return new ComboIntAttribute( combo, value, parent );
    }

    static ComboIntAttribute* bindSlot( QComboBox *combo, QObject *object, const char *slot )
    {
        ComboIntAttribute *attr = new ComboIntAttribute( combo, NULL, object );
        assert( connect(attr, SIGNAL(attributeChanged(int)), object, slot) );
        return attr;
    }

    static ComboIntAttribute* bindIntAndSlot( QComboBox *combo, int *value, QObject *object, const char *slot )
    {
        ComboIntAttribute *attr = new ComboIntAttribute( combo, value, object );
        assert( connect(attr, SIGNAL(attributeChanged(int)), object, slot) );
        return attr;
    }

signals:

    void attributeChanged( int value );

public slots:

    void valueChanged( int value )
    {
        bool shouldEmit = true;
        if ( m_value ) {
            if ( (shouldEmit = (*m_value != value)) ) {
                *m_value = value;
            }
        }
        m_combo->setCurrentIndex( value );
        if ( shouldEmit ) emit attributeChanged( value );
    }

private:

    QComboBox *m_combo;

    int *m_value;


};

#endif // DATABINDING_H
