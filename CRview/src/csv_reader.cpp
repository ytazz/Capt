#include "../include/csv_reader.h"

vector<Data> CSVReader::getData()
{
  FILE *fp;
  errno_t err;
  err = fopen_s(&fp, fileName.c_str(), "r");

  if (err != 0) {
    cout << "file " <<  fileName.c_str() << " cannot be opened!" << endl;
    exit(EXIT_FAILURE);

  }else{
    vector<Data> datalist;
    Data buf;
    int count = 0;

    cout << "Reading file" << endl;
    while (readLine(fp, &buf) != EOF)
    {
      datalist.push_back(buf);
      count++;
    }

    fclose(fp);

    cout << "finished :" << count << endl;

    return datalist;
  }
}

int CSVReader::readLine(FILE *fp, Data* pbuf)
{
  return fscanf_s(fp, "%f, %f, %f, %f, %d, %f, %f, %d",
                  &pbuf->state.icp.r,
                  &pbuf->state.icp.th,
                  &pbuf->state.swf.r,
                  &pbuf->state.swf.th,
                  &pbuf->state.n,
                  &pbuf->input.dsf.r,
                  &pbuf->input.dsf.th,
                  &pbuf->input.n);
}
