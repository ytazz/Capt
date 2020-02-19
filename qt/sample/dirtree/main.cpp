#include <QApplication>
#include <QVBoxLayout>
#include <QTreeView>

class WidgetHierarchyModel : public QAbstractItemModel
{
   public:
    WidgetHierarchyModel(QWidget* topWidget):topWidget(topWidget)
    {
    }

    QWidget* widget(const QModelIndex &index) const
    {
        return static_cast<QWidget*>(index.internalPointer());
    }

    QVariant data(const QModelIndex &index, int role) const override
    {
        if( role != Qt::DisplayRole || !index.isValid() ) return QVariant();
        return widget(index)->metaObject()->className();
    }

    QVariant headerData(int, Qt::Orientation orientation, int role) const override
    {
        if( orientation != Qt::Horizontal || role != Qt::DisplayRole ) return QVariant();
        return QString("Widget Hierarchy");
    }

    int rowCount(const QModelIndex &parent) const override
    {
        return parent.isValid() ? childrenOf(widget(parent)).size() : 1;
    }

    int columnCount(const QModelIndex &) const override
    {
        return 1;
    }

    QModelIndex index(int row, int column, const QModelIndex &parent) const override
    {
        if( !parent.isValid() )
        {
            if( row == 0 && column == 0 )
            {
                return createIndex(0, 0, topWidget);
            }
            return QModelIndex();
        }
        if( column != 0 || parent.column() != 0 )
        {
            return QModelIndex();
        }
        QList<QWidget*> children = childrenOf(widget(parent));
        if( row < children.size() )
        {
            return createIndex(row, 0, children.at(row));
        }
        return QModelIndex();
    }

    QModelIndex parent(const QModelIndex &index) const override
    {
        if( index.isValid() )
        {
            QWidget* const self = widget(index);
            if( self != topWidget )
            {
                QWidget* const parent = self->parentWidget();
                int row = childrenOf(parent).indexOf(self);
                if( row > -1 )
                {
                    return createIndex(row, 0, parent);
                }
            }
        }
        return QModelIndex();
    }

  private:
    static QList<QWidget*> childrenOf(const QWidget* parent)
    {
        return parent->findChildren<QWidget*>(QString(), Qt::FindDirectChildrenOnly);
    }

    QWidget* topWidget;
};

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    QWidget* window = new QWidget();
    QVBoxLayout* layout = new QVBoxLayout();
    QTreeView* tree   = new QTreeView(window);

    layout->addWidget(tree);
    window->setLayout(layout);

    WidgetHierarchyModel* model = new WidgetHierarchyModel(tree);
    tree->setModel(model);
    tree->setHeaderHidden(false);
    window->show();

    a.exec();
    delete window;

    return 0;
}
